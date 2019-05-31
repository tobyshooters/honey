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
#include "media/homogeneous.h"

namespace pbrt {

void PhotonIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {
    chooseLightDistribution = ComputeLightPowerDistribution(scene);
    // TODO: other distributions? Add to CreatePhotonIntegrator if need be
    sampleLightDistribution = CreateLightSampleDistribution("spatial", scene);

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
        int lightNum = chooseLightDistribution->SampleDiscrete(lightSample, &lightPdf);
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

            // Handle an interaction with a medium or a surface
            if (mi.IsValid()) {
                // Terminate path if ray escaped or _maxDepth_ was reached
                if (bounces >= maxDepth) break;

                // TODO: factor in power
                if (bounces > 0) {
                  // Break into many beams with length based on beamLength
                  float beamTime = beamLength / (photonRay.d.Length());

                  // TODO: If breaks, check if isect.time is correct
                  for (float startTime = 0; startTime < isect.time; startTime += beamTime) { 
                    float endTime = std::min(startTime + beamTime, isect.time);
                    beams.push_back(std::make_shared<Beam>(
                          photonRay, startTime, endTime, /*power =*/ 0, initialSearchRadius));
                  }
                }

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
                
                // TODO: possibly consider incorporating BSSRDF for photon tracing

            }

            // TODO: terminate path by russian roulette
        }
    }

    bvh = std::make_shared<BVHAccel>(beams);
}

Spectrum PhotonIntegrator::Li(const RayDifferential &r, const Scene &scene,
        Sampler &sampler, MemoryArena &arena, int depth) const
{
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f), beta(1.f);
    RayDifferential ray(r);
    bool specularBounce = false;
    int bounces;

    for (bounces = 0;; ++bounces) {
        // Intersect _ray_ with scene and store intersection in _isect_
        SurfaceInteraction isect;
        bool foundIntersection = scene.Intersect(ray, &isect);

        // Sample the participating medium, if present
        MediumInteraction mi;
        if (ray.medium) beta *= ray.medium->Sample(ray, sampler, arena, &mi);
        if (beta.IsBlack()) break;

        // Handle an interaction with a medium or a surface
        if (mi.IsValid()) {
            // Terminate path if ray escaped or _maxDepth_ was reached
            if (bounces >= maxDepth) break;

            std::vector<BeamInteraction*> isects;
            bvh->AllIntersects(ray, isects);

            float photonContribution;

            // Do Ray x Beam (1D) intersection
            for (BeamInteraction* i : isects) {

                // TODO: block usage of heterogeneous media

                HomogeneousMedium* hm = (HomogeneousMedium*) ray.medium;
                Float phase = PhaseHG(i->bi.theta, hm->g);

                /* Spectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) * ray.d.Length()); */
            }

            L += beta * photonContribution;
            break;
        } else {
            // Handle scattering at point on surface for volumetric path tracer

            // Possibly add emitted light at intersection
            if (bounces == 0 || specularBounce) {
                // Add emitted light at path vertex or from the environment
                if (foundIntersection)
                    L += beta * isect.Le(-ray.d);
                else
                    for (const auto &light : scene.infiniteLights)
                        L += beta * light->Le(ray);
            }

            // Terminate path if ray escaped or _maxDepth_ was reached
            if (!foundIntersection || bounces >= maxDepth) break;

            // Compute scattering functions and skip over medium boundaries
            isect.ComputeScatteringFunctions(ray, arena, true);
            if (!isect.bsdf) {
                ray = isect.SpawnRay(ray.d);
                bounces--;
                continue;
            }

            // Sample illumination from lights to find attenuated path
            // contribution
            const Distribution1D *lightDistrib = sampleLightDistribution->Lookup(isect.p);
            L += beta * UniformSampleOneLight(isect, scene, arena, sampler,
                                              true, lightDistrib);

            // Sample BSDF to get new path direction
            Vector3f wo = -ray.d, wi;
            Float pdf;
            BxDFType flags;
            Spectrum f = isect.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdf,
                                              BSDF_ALL, &flags);
            if (f.IsBlack() || pdf == 0.f) break;
            beta *= f * AbsDot(wi, isect.shading.n) / pdf;
            DCHECK(std::isinf(beta.y()) == false);
            specularBounce = (flags & BSDF_SPECULAR) != 0;
            ray = isect.SpawnRay(wi);

            // Account for attenuated subsurface scattering, if applicable
            if (isect.bssrdf && (flags & BSDF_TRANSMISSION)) {
                // Importance sample the BSSRDF
                SurfaceInteraction pi;
                Spectrum S = isect.bssrdf->Sample_S(
                    scene, sampler.Get1D(), sampler.Get2D(), arena, &pi, &pdf);
                DCHECK(std::isinf(beta.y()) == false);
                if (S.IsBlack() || pdf == 0) break;
                beta *= S / pdf;

                // Account for the attenuated direct subsurface scattering
                // component
                L += beta *
                     UniformSampleOneLight(pi, scene, arena, sampler, true,
                                           sampleLightDistribution->Lookup(pi.p));

                // Account for the indirect subsurface scattering component
                Spectrum f = pi.bsdf->Sample_f(pi.wo, &wi, sampler.Get2D(),
                                               &pdf, BSDF_ALL, &flags);
                if (f.IsBlack() || pdf == 0) break;
                beta *= f * AbsDot(wi, pi.shading.n) / pdf;
                DCHECK(std::isinf(beta.y()) == false);
                specularBounce = (flags & BSDF_SPECULAR) != 0;
                ray = pi.SpawnRay(wi);
            }
        }

        // Possibly terminate the path with Russian roulette
        // Factor out radiance scaling due to refraction in rrBeta.
        float rrThreshold = 1.0;
        if (beta.MaxComponentValue() < rrThreshold && bounces > 3) {
            Float q = std::max((Float).05, 1 - beta.MaxComponentValue());
            if (sampler.Get1D() < q) break;
            beta /= 1 - q;
            DCHECK(std::isinf(beta.y()) == false);
        }
    }
    return L;
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
