
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_PRIMITIVE_H
#define PBRT_CORE_PRIMITIVE_H

// core/primitive.h*
#include "pbrt.h"
#include "shape.h"
#include "material.h"
#include "medium.h"
#include "transform.h"
#include "interaction.h"

#define print(x) std::cout << x << std::endl;

namespace pbrt {

// Primitive Declarations
class Primitive {
  public:
    // Primitive Interface
    virtual ~Primitive();
    virtual Bounds3f WorldBound() const = 0;
    virtual bool Intersect(const Ray &r, SurfaceInteraction *) const = 0;
    virtual bool IntersectP(const Ray &r) const = 0;
    virtual const AreaLight *GetAreaLight() const = 0;
    virtual const Material *GetMaterial() const = 0;
    virtual void ComputeScatteringFunctions(SurfaceInteraction *isect,
                                            MemoryArena &arena,
                                            TransportMode mode,
                                            bool allowMultipleLobes) const = 0;
};

// GeometricPrimitive Declarations
class GeometricPrimitive : public Primitive {
  public:
    // GeometricPrimitive Public Methods
    virtual Bounds3f WorldBound() const;
    virtual bool Intersect(const Ray &r, SurfaceInteraction *isect) const;
    virtual bool IntersectP(const Ray &r) const;
    GeometricPrimitive(const std::shared_ptr<Shape> &shape,
                       const std::shared_ptr<Material> &material,
                       const std::shared_ptr<AreaLight> &areaLight,
                       const MediumInterface &mediumInterface);
    const AreaLight *GetAreaLight() const;
    const Material *GetMaterial() const;
    void ComputeScatteringFunctions(SurfaceInteraction *isect,
                                    MemoryArena &arena, TransportMode mode,
                                    bool allowMultipleLobes) const;

  private:
    // GeometricPrimitive Private Data
    std::shared_ptr<Shape> shape;
    std::shared_ptr<Material> material;
    std::shared_ptr<AreaLight> areaLight;
    MediumInterface mediumInterface;
};

// TransformedPrimitive Declarations
class TransformedPrimitive : public Primitive {
  public:
    // TransformedPrimitive Public Methods
    TransformedPrimitive(std::shared_ptr<Primitive> &primitive,
                         const AnimatedTransform &PrimitiveToWorld);
    bool Intersect(const Ray &r, SurfaceInteraction *in) const;
    bool IntersectP(const Ray &r) const;
    const AreaLight *GetAreaLight() const { return nullptr; }
    const Material *GetMaterial() const { return nullptr; }
    void ComputeScatteringFunctions(SurfaceInteraction *isect,
                                    MemoryArena &arena, TransportMode mode,
                                    bool allowMultipleLobes) const {
        LOG(FATAL) <<
            "TransformedPrimitive::ComputeScatteringFunctions() shouldn't be "
            "called";
    }
    Bounds3f WorldBound() const {
        return PrimitiveToWorld.MotionBounds(primitive->WorldBound());
    }

  private:
    // TransformedPrimitive Private Data
    std::shared_ptr<Primitive> primitive;
    const AnimatedTransform PrimitiveToWorld;
};

// Aggregate Declarations
class Aggregate : public Primitive {
  public:
    // Aggregate Public Methods
    const AreaLight *GetAreaLight() const;
    const Material *GetMaterial() const;
    void ComputeScatteringFunctions(SurfaceInteraction *isect,
                                    MemoryArena &arena, TransportMode mode,
                                    bool allowMultipleLobes) const;
};

class Beam : public Primitive {
    public:
        Beam(Ray &r, Float tStart, Float tEnd, Spectrum power, Float radius)
            : r(r), tStart(tStart), tEnd(tEnd), power(power), radius(radius) {}

        // TODO: check if we need to be updating query.tMax to intersect
        bool Intersect(const Ray& query, SurfaceInteraction* isect) const {

            // Express origin and direction
            Vector3f a = Vector3f(query.o);
            Vector3f b = query.d;
            Vector3f c = Vector3f(r.o);
            Vector3f d = r.d;

            // std::cout << "Beam length: " << Distance(r(tStart), r(tEnd)) << std::endl;

            // Solve quadratic
            Float t = (Dot(c - a, b) + (Dot(a - c, d) * Dot(b,d))) / (1 - (Dot(b,d) * Dot(b,d)));
            Float s = (-1.f * Dot(c - a, d) + (Dot(a - c, d) * Dot(b,d))) / (1 - (Dot(b,d) * Dot(b,d)));

            if (s < tStart || s > tEnd) {
                return false;
            }
            
            Float dist = Distance(query(t), r(s));

            // Abuse isect to store relevant information
            if (isect) {
                ((BeamInteraction*) isect)->bi = { 
                  r, s, t, Dot(r.d, query.d), power, radius
                };
            }
            // std::cout << "Radius: " << radius << "Dist: " << dist << std::endl;
            // Return hit
            return dist < radius;
        }

        // TODO: make sure this IntersectP does not need to do anything specific
        bool IntersectP(const Ray& query) const { return Intersect(query, nullptr); }
        Bounds3f WorldBound()             const { return Bounds3f(r(tStart), r(tEnd)); }
        const AreaLight* GetAreaLight()   const { return nullptr; }
        const Material* GetMaterial()     const { return nullptr; }
        void ComputeScatteringFunctions(
            SurfaceInteraction* i, MemoryArena& a, TransportMode m, bool b) const { return; }

    private:
        Ray r; // origin, direction, medium, tMax (set to end of medium)
        Float tStart, tEnd;
        Spectrum power;
        Float radius;
};


}  // namespace pbrt

#endif  // PBRT_CORE_PRIMITIVE_H
