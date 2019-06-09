# Honey
Rendering honey for the CS 348B Image Synthesis Techniques rendering competition

Relevant files:
- `src/integrators/photon.cpp`: photon beam integration implementation
- `src/accelerators/bvh.cpp`: bvh extension for phootn beam, AllIntersects implementation
- `src/core/interaction.h`: BeamInteraction class extending Interaction
- `src/core/primitive.h`: Beam with ray-beam intersection that can be stored in BVH

Changes from PBRT:
```
 src/accelerators/bvh.cpp    |  49 ++++++++++...
 src/accelerators/bvh.h      |   1 +
 src/core/api.cpp            |   5 ++-
 src/core/interaction.h      |  19 ++++++++++...
 src/core/primitive.h        |  56 ++++++++++...
 src/integrators/photon.cpp  | 356 ++++++++++...
 src/integrators/photon.h    |  65 ++++++++++...
 src/media/homogeneous.h     |   3 +-
```
