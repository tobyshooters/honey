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

#define print(x) std::cout << x << std::endl;

namespace pbrt {

void PhotonIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {
    Float beamRadius = initialSearchRadius;
    chooseLightDistribution = ComputeLightPowerDistribution(scene);
    // TODO: other distributions? Add to CreatePhotonIntegrator if need be
    sampleLightDistribution = CreateLightSampleDistribution("spatial", scene);

    MemoryArena arena;
    std::vector<std::shared_ptr<Primitive>> beams;

    // Should have a separate sampler for this part
    HaltonSampler preSampler(photonsPerIteration, camera->film->GetSampleBounds());  

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
        Point2f uLight0(RadicalInverse(haltonDim,     haltonIndex), RadicalInverse(haltonDim + 1, haltonIndex));
        Point2f uLight1(RadicalInverse(haltonDim + 2, haltonIndex), RadicalInverse(haltonDim + 3, haltonIndex));
        Float uLightTime = Lerp(RadicalInverse(haltonDim + 4, haltonIndex), camera->shutterOpen, camera->shutterClose);
        haltonDim += 5;

        // Generate _photonRay_ from light source and initialize _beta_
        RayDifferential photonRay;
        Normal3f nLight;
        Float pdfPos, pdfDir;
        Spectrum Le = light->Sample_Le(uLight0, uLight1, uLightTime, &photonRay, &nLight, &pdfPos, &pdfDir);

        if (pdfPos == 0 || pdfDir == 0 || Le.IsBlack()) continue;
        Spectrum beta = (AbsDot(nLight, photonRay.d) * Le) / (lightPdf * pdfPos * pdfDir);
        if (beta.IsBlack()) continue;

        bool specularBounce = false;

        int bounces;
        for (bounces = 0;; ++bounces) {
            // Intersect _ray_ with scene and store intersection in _isect_
            SurfaceInteraction isect;
            bool foundIntersection = scene.Intersect(photonRay, &isect);

            // Sample the participating medium, if present
            MediumInteraction mi;
            if (photonRay.medium) beta *= photonRay.medium->Sample(photonRay, preSampler, arena, &mi);
            if (beta.IsBlack()) break;
            photonRay.d = Normalize(photonRay.d);

            // IMPORTANT:
            // ----------
            // We store photon beams if they are inside of medium
            // Outside of medium, indirect illumination is calculated per VolPath

            // Handle an interaction with a medium or a surface
            if (mi.IsValid()) {
                // Terminate path if ray escaped or _maxDepth_ was reached
                if (bounces >= maxPhotonDepth) break;

                if (bounces > 0) {

                    // Break into many beams with length based on beamLength
                    float beamTime = beamLength;

                    // TODO: If breaks, check if isect.time is correct
                    for (float startTime = 0; startTime < isect.time; startTime += beamTime) { 
                        float endTime = std::min(startTime + beamTime, isect.time);
                        beams.push_back(std::make_shared<Beam>(
                                    photonRay, startTime, endTime, /*power =*/ beta, beamRadius));
                    }
                }

                Vector3f wo = -photonRay.d, wi; 
                mi.phase->Sample_p(wo, &wi, preSampler.Get2D());
                photonRay = mi.SpawnRay(wi);
                specularBounce = false;
            } else {

                // Terminate path if ray escaped or _maxDepth_ was reached
                if (!foundIntersection || bounces >= maxPhotonDepth) break;

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
                Spectrum f = isect.bsdf->Sample_f(wo, &wi, preSampler.Get2D(), &pdf,
                                                  BSDF_ALL, &flags);
                if (f.IsBlack() || pdf == 0.f) break;

                beta *= f * AbsDot(wi, isect.shading.n) / pdf;
                DCHECK(std::isinf(beta.y()) == false);

                specularBounce = (flags & BSDF_SPECULAR) != 0;
                photonRay = isect.SpawnRay(wi);
                
                // TODO: possibly consider incorporating BSSRDF for photon tracing

            }

            // TODO: terminate path by russian roulette
        }
        preSampler.StartNextSample();
    }

    bvh = std::make_shared<BVHAccel>(beams);
}

Spectrum PhotonIntegrator::Li(const RayDifferential &r, const Scene &scene,
        Sampler &sampler, MemoryArena &arena, int depth) const
{
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f), beta(1.f);
    RayDifferential ray(r);
    Float beamRadius = initialSearchRadius;

    bool specularBounce = false;
    int bounces;

    //print("--------------------")

    for (bounces = 0;; ++bounces) {

        // Just making sure...
        ray.d = Normalize(ray.d);

        //print("bounce: " + std::to_string(bounces))

        // Intersect _ray_ with scene and store intersection in _isect_
        SurfaceInteraction isect;
        bool foundIntersection = scene.Intersect(ray, &isect);

        //print("intersected")

        // Sample the participating medium, if present
        MediumInteraction mi;
        if (ray.medium) beta *= ray.medium->Sample(ray, sampler, arena, &mi);
        if (beta.IsBlack()) break;

        //print("sampled medium")

        // Handle an interaction with a medium or a surface
        if (mi.IsValid()) {
            // Terminate path if ray escaped or _maxDepth_ was reached
            if (bounces >= maxDepth) break;

            //print("in medium")

            std::vector<BeamInteraction*> isects;
            bvh->AllIntersects(ray, isects);

            // TODO: block usage of heterogeneous media
            HomogeneousMedium* hm = (HomogeneousMedium*) ray.medium;
            Spectrum photonContribution = Spectrum(0);

            // Do Ray x Beam (1D) intersection
            for (BeamInteraction* i : isects) {

                //print("looping isects")

                Float phase = PhaseHG(i->bi.cosTheta, hm->g);
                Spectrum Tr_t = Exp(-hm->sigma_t * std::min(i->bi.t, MaxFloat));
                Spectrum Tr_s = Exp(-hm->sigma_t * std::min(i->bi.s, MaxFloat));

                Spectrum photonThroughput = phase * Tr_t * Tr_s / std::sqrt(1 - (i->bi.cosTheta * i->bi.cosTheta));
                photonContribution += photonThroughput * i->bi.power;

                delete i;
            }

            //print("found isects")

            L += beta * hm->sigma_s * (1 / beamRadius) * photonContribution;

            // FINAL GATHERING
            for (int fg = 0; fg < 10; fg ++) {

                Vector3f wo = -ray.d, wi;
                mi.phase->Sample_p(wo, &wi, sampler.Get2D());
                Ray rayFg = mi.SpawnRay(wi);

                MediumInteraction miFg;
                if (ray.medium) beta *= ray.medium->Sample(ray, sampler, arena, &miFg);
                if (beta.IsBlack()) break;

                if (miFg.IsValid()) {
                    std::vector<BeamInteraction*> isectsFg;
                    bvh->AllIntersects(rayFg, isectsFg);

                    HomogeneousMedium* hmFg = (HomogeneousMedium*) rayFg.medium;
                    Spectrum photonFgContribution = Spectrum(0);

                    for (BeamInteraction* i : isectsFg) {
                        Float phase = PhaseHG(i->bi.cosTheta, hm->g);
                        Spectrum Tr_t = Exp(-hmFg->sigma_t * std::min(i->bi.t, MaxFloat));
                        Spectrum Tr_s = Exp(-hmFg->sigma_t * std::min(i->bi.s, MaxFloat));

                        Spectrum photonThroughput = phase * Tr_t * Tr_s / std::sqrt(1 - (i->bi.cosTheta * i->bi.cosTheta));
                        photonFgContribution += photonThroughput * i->bi.power;

                        delete i;
                    }
                    L += beta * hmFg->sigma_s * (1 / beamRadius) * photonFgContribution;
                }
            }

            break;

        } else {
            // Handle scattering at point on surface for volumetric path tracer
            //print("non-volumetric bounce")

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

            //print("sampled direct lights")

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

                //print("sampled bssrdf")

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

        //print("russian roulette'd")

        float rrThreshold = 1.0;
        if (beta.MaxComponentValue() < rrThreshold && bounces > 3) {
            Float q = std::max((Float).05, 1 - beta.MaxComponentValue());
            if (sampler.Get1D() < q) break;
            beta /= 1 - q;
            DCHECK(std::isinf(beta.y()) == false);
        }
    }
    //print("calculated L")
    return L;
}

Integrator *CreatePhotonIntegrator(
        const ParamSet &params, std::shared_ptr<Sampler> sampler,
        std::shared_ptr<const Camera> camera)
{
    //print(camera);
    int nIterations = params.FindOneInt("iterations", params.FindOneInt("numiterations", 64));
    int maxPhotonDepth = params.FindOneInt("maxphotondepth", 5);
    int maxDepth = params.FindOneInt("maxdepth", 5);
    int photonsPerIter = params.FindOneInt("photonsperiteration", -1);
    int writeFreq = params.FindOneInt("imagewritefrequency", 1 << 31);
    Float radius = params.FindOneFloat("radius", 1.f);
    if (PbrtOptions.quickRender) nIterations = std::max(1, nIterations / 16);

    return new PhotonIntegrator(camera, sampler, nIterations, photonsPerIter,
                                maxDepth, maxPhotonDepth, radius, writeFreq);
}

}
