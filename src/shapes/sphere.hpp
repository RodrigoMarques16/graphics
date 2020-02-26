#pragma once

#include "hit.hpp"
#include "rgb.hpp"
#include "ray.hpp"
#include "vec3.hpp"
#include <optional>

struct Sphere {
    Vec3 center;
    double radiusSquared;

    constexpr Sphere(const Vec3& c, double r) : center(c), radiusSquared(r) {}
};