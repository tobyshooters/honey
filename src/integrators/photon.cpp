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
        int bounces;
        for (bounces = 0;; ++bounces) {
            // Intersect _ray_ with scene and store intersection in _isect_
            SurfaceInteraction isect;
            bool foundIntersection = scene.Intersect(photonRay, &isect);

            // Sample the participating medium, if present
            MediumInteraction mi;
            if (photonRay.medium) beta *= photonRay.medium->Sample(photonRay, sampler, arena, &mi);
            if (beta.IsBlack()) break;

            // Handle an interaction with a medium or a surface
            if (mi.IsValid()) {
                // Terminate path if ray escaped or _maxDepth_ was reached
                if (bounces >= maxDepth) break;

                if (bounces > 0) {
                    // Add to beam to primitives
                    // TODO: break into many beams
                    // TODO: factor in power
                    beams.push_back(std::make_shared<Beam>(
                                photonRay, 0, photonRay.tMax, /*power =*/ 0, initialSearchRadius));
                }

                Vector3f wo = -photonRay.d, wi; mi.phase->Sample_p(wo, &wi, sampler.Get2D());
                photonRay = mi.SpawnRay(wi);
            } else {

                // Store beam if on diffuse
            }
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
