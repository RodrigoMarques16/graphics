#pragma once

#include "hit.hpp"
#include "ray.hpp"
#include "vec3.hpp"
#include <optional>

struct Triangle {
    Vec3 p0, p1, p2;

    constexpr Triangle(const Vec3 &a, const Vec3 &b, const Vec3 &c)
        : p0(a), p1(b), p2(c) {}

    constexpr Triangle(const Triangle &other) = default;
};