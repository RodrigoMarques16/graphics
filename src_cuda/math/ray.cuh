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
    __device__ Hit() {};
    __device__ Hit(double dist, rgb color) : dist(dist), color(color) {}
};

struct ShadowHit {
    double dist;
    __device__ ShadowHit() {};
    __device__ ShadowHit(double dist) : dist(dist) {}
};