#pragma once

#include "../math/vec3.cuh"

struct Sphere {
    Vec3 center;
    double radiusSquared;

    Sphere() = default;

    constexpr Sphere(const Vec3& c, double r) : center(c), radiusSquared(r) {}
};