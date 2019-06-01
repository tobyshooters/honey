#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_INTEGRATORS_PHOTON_H
#define PBRT_INTEGRATORS_PHOTON_H

// integrators/photon.h*
#include "pbrt.h"
#include "integrator.h"
#include "camera.h"
#include "film.h"
#include "accelerators/bvh.h"
#include "lightdistrib.h"

namespace pbrt {

// VolSPPM Declarations
class PhotonIntegrator : public SamplerIntegrator {
  public:
    PhotonIntegrator(std::shared_ptr<const Camera> &camera, 
                     std::shared_ptr<Sampler> sampler,
                     int nIterations, int photonsPerIteration, int maxDepth,
                     Float initialSearchRadius, int writeFrequency)
        : SamplerIntegrator(camera, sampler, camera->film->GetSampleBounds()),
          initialSearchRadius(initialSearchRadius),
          nIterations(nIterations),
          maxDepth(maxDepth),
          photonsPerIteration(photonsPerIteration > 0
                              ? photonsPerIteration
                              : camera->film->croppedPixelBounds.Area()),
          writeFrequency(writeFrequency) {}

    void Preprocess(const Scene &scene, Sampler &sampler);

    Spectrum Li(const RayDifferential &ray, const Scene &scene,
                Sampler &sampler, MemoryArena &arena, int depth) const;

  private:
    // std::shared_ptr<const Camera> camera; // already in SamplerIntegrator
    const Float initialSearchRadius;
    const int nIterations;
    const int maxDepth;
    const int photonsPerIteration;
    const int writeFrequency;

    std::unique_ptr<Distribution1D> chooseLightDistribution;
    std::unique_ptr<LightDistribution> sampleLightDistribution;
    std::shared_ptr<BVHAccel> bvh;
};

Integrator *CreatePhotonIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera);

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_PHOTON_H
