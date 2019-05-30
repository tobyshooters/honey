// integrators/sppm.cpp*
#include "integrators/photon.h"
#include "parallel.h"
#include "scene.h"
#include "imageio.h"
#include "spectrum.h"
#include "rng.h"
#include "paramset.h"
#include "progressreporter.h"
#include "interaction.h"
#include "sampling.h"
#include "samplers/halton.h"
#include "stats.h"
#include "accelerators/bvh.h"
#include "lightdistrib.h"
#include "primitive.h"
#include "bssrdf.h"

namespace pbrt {

class Beam : public Primitive {
    public:
        Beam(Ray &r, Float tStart, Float tEnd, Float power, Float radius)
            : r(r), tStart(tStart), tEnd(tEnd), power(power), radius(radius) {}

        bool Intersect(const Ray& query, SurfaceInteraction* isect) const {
            Vector3f v = Cross(r.d, query.d);

            // Express origin and direction
            Vector3f a = Vector3f(query.o);
            Vector3f b = query.d;
            Vector3f c = Vector3f(r.o);
            Vector3f d = r.d;

            // Solve quadratic
            Float t = (Dot(b, d) * (Dot(c, d) - Dot(a, d)) - (Dot(b, c) * Dot(a, b))) 
                / (Dot(b, d) * Dot(b ,d) - 1);

            Float s = (Dot(b, d) * (Dot(a, d) - Dot(b, c)) - (Dot(a, d) * Dot(c, d))) 
                / (Dot(b, d) * Dot(b ,d) - 1);
            
            Float dist = Distance(query(t), r(s));

            // Abuse isect to store relevant information
            if (isect)
                isect->p = Point3f(dist, s, t);

            // Return hit
            return dist < radius;
        }

        bool IntersectP(const Ray& query) const { return Intersect(query, nullptr); }
        Bounds3f WorldBound()             const { return Bounds3f(r(tStart), r(tEnd)); }
        const AreaLight* GetAreaLight()   const { return nullptr; }
        const Material* GetMaterial()     const { return nullptr; }
        void ComputeScatteringFunctions(
                SurfaceInteraction* i, MemoryArena& a, TransportMode m, bool b) const { return; }

    private:
        Ray r; // origin, direction, medium, tMax (set to end of medium)
        Float tStart, tEnd;
        Float power;
        Float radius;
};

void PhotonIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {
    lightDistribution = ComputeLightPowerDistribution(scene);
    MemoryArena arena;
    std::vector<std::shared_ptr<Primitive>> beams;

    // Declare a beam segment size to break the whole thing up into
    float beamLength = 0.05f;

    for (int photonIndex = 0; photonIndex < photonsPerIteration; photonIndex++) {
        // Follow photon path for _photonIndex_
        int iter = 0;
        uint64_t haltonIndex = (uint64_t)iter * (uint64_t)photonsPerIteration + photonIndex;
        int haltonDim = 0;

        // Choose light to shoot photon from
        Float lightPdf;
        Float lightSample = RadicalInverse(haltonDim++, haltonIndex);
        int lightNum = lightDistribution->SampleDiscrete(lightSample, &lightPdf);
        const std::shared_ptr<Light> &light = scene.lights[lightNum];

        // Compute sample values for photon ray leaving light source
        Point2f uLight0(RadicalInverse(haltonDim,     haltonIndex),
                RadicalInverse(haltonDim + 1, haltonIndex));
        Point2f uLight1(RadicalInverse(haltonDim + 2, haltonIndex),
                RadicalInverse(haltonDim + 3, haltonIndex));
        Float uLightTime =
            Lerp(RadicalInverse(haltonDim + 4, haltonIndex),
                    camera->shutterOpen, camera->shutterClose);
        haltonDim += 5;

        // Generate _photonRay_ from light source and initialize _beta_
        RayDifferential photonRay;
        Normal3f nLight;
        Float pdfPos, pdfDir;
        Spectrum Le = light->Sample_Le(uLight0, uLight1, uLightTime, &photonRay,
                &nLight, &pdfPos, &pdfDir);
        if (pdfPos == 0 || pdfDir == 0 || Le.IsBlack()) return;
        Spectrum beta = (AbsDot(nLight, photonRay.d) * Le) / (lightPdf * pdfPos * pdfDir);
        if (beta.IsBlack()) return;

        /* Spectrum power = light->Power() / photonsPerIteration; */

        // TODO: use russian roulette
        bool specularBounce = false;

        int bounces;
        for (bounces = 0;; ++bounces) {
            // Intersect _ray_ with scene and store intersection in _isect_
            SurfaceInteraction isect;
            bool foundIntersection = scene.Intersect(photonRay, &isect);

            // Sample the participating medium, if present
            MediumInteraction mi;
            // TODO: power check here
            if (photonRay.medium) beta *= photonRay.medium->Sample(photonRay, sampler, arena, &mi);
            if (beta.IsBlack()) break;

            // IMPORTANT:
            // ----------
            // We store photon beams if they are inside of medium
            // Outside of medium, indirect illumination is calculated per VolPath

            // Store beam if on diffuse
            if (bounces > 0) {
                // Add to beam to primitives
                // TODO: factor in power
                // Break into many beams with length based on beamLength
                float beamTime = beamLength / (photonRay.d.Length());
                // TODO: how to loop to end of medium?
                for (float startTime = 0; startTime < mi.time; startTime += beamTime) { 
                    float endTime = std::min(startTime + beamTime, mi.time);
                    beams.push_back(std::make_shared<Beam>(
                                photonRay, startTime, endTime, /*power =*/ 0, initialSearchRadius));
                }
            }

            // Handle an interaction with a medium or a surface
            if (mi.IsValid()) {
                // Terminate path if ray escaped or _maxDepth_ was reached
                if (bounces >= maxDepth) break;

                Vector3f wo = -photonRay.d, wi; 
                mi.phase->Sample_p(wo, &wi, sampler.Get2D());
                photonRay = mi.SpawnRay(wi);
                specularBounce = false;
            } else {

                // Terminate path if ray escaped or _maxDepth_ was reached
                if (!foundIntersection || bounces >= maxDepth) break;

                // Compute scattering functions and skip over medium boundaries
                // Think this is a "None" material
                isect.ComputeScatteringFunctions(photonRay, arena, true);
                if (!isect.bsdf) {
                    photonRay = isect.SpawnRay(photonRay.d);
                    bounces--;
                    continue;
                }

                // Sample BSDF to get new path direction
                Vector3f wo = -photonRay.d, wi;
                Float pdf;
                BxDFType flags;
                Spectrum f = isect.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdf,
                                                  BSDF_ALL, &flags);
                if (f.IsBlack() || pdf == 0.f) break;

                // TODO: attenuate power by BSDF
                beta *= f * AbsDot(wi, isect.shading.n) / pdf;
                DCHECK(std::isinf(beta.y()) == false);

                specularBounce = (flags & BSDF_SPECULAR) != 0;
                photonRay = isect.SpawnRay(wi);

                // Account for attenuated subsurface scattering, if applicable
                if (isect.bssrdf && (flags & BSDF_TRANSMISSION)) {
                    // Importance sample the BSSRDF
                    SurfaceInteraction pi;
                    Spectrum S = isect.bssrdf->Sample_S(
                            scene, sampler.Get1D(), sampler.Get2D(), arena, &pi, &pdf);
                    DCHECK(std::isinf(beta.y()) == false);
                    if (S.IsBlack() || pdf == 0) break;
                    // TODO: attenuate power via BSSRDF
                    beta *= S / pdf;

                    // Account for the indirect subsurface scattering component
                    Spectrum f = pi.bsdf->Sample_f(pi.wo, &wi, sampler.Get2D(),
                            &pdf, BSDF_ALL, &flags);
                    if (f.IsBlack() || pdf == 0) break;
                    // TODO: attenuate power via BSSRDF
                    beta *= f * AbsDot(wi, pi.shading.n) / pdf;
                    DCHECK(std::isinf(beta.y()) == false);
                    specularBounce = (flags & BSDF_SPECULAR) != 0;
                    photonRay = pi.SpawnRay(wi);
                }
            }

            // TODO: terminate path by russian roulette
        }
    }

    bvh = std::make_shared<BVHAccel>(beams);
}

Spectrum PhotonIntegrator::Li(const RayDifferential &ray, const Scene &scene,
        Sampler &sampler, MemoryArena &arena, int depth) const
{

    // Shoot Camera Rays
        // if specular
        // if diffuse
      
    return Spectrum(0);
}

Integrator *CreatePhotonIntegrator(
        const ParamSet &params, std::shared_ptr<Sampler> sampler,
        std::shared_ptr<const Camera> camera)
{
    int nIterations = params.FindOneInt("iterations", params.FindOneInt("numiterations", 64));
    int maxDepth = params.FindOneInt("maxdepth", 5);
    int photonsPerIter = params.FindOneInt("photonsperiteration", -1);
    int writeFreq = params.FindOneInt("imagewritefrequency", 1 << 31);
    Float radius = params.FindOneFloat("radius", 1.f);
    if (PbrtOptions.quickRender) nIterations = std::max(1, nIterations / 16);

    return new PhotonIntegrator(camera, sampler, nIterations, photonsPerIter,
                                maxDepth, radius, writeFreq);
}

}
