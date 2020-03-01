#pragma once

#include "vec3.cuh"
#include "rgb.cuh"

struct Ray {
    Vec3 origin, direction;

    __device__ constexpr Vec3 at(double t) const {
        return origin + t * direction;
    }
};

struct Hit {
    double dist;
    rgb color;
};

struct ShadowHit {
    double dist;
};