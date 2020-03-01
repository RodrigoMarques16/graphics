#pragma once

#include "../math/vec3.cuh"

struct Triangle {
    Vec3 p0, p1, p2;

    Triangle() = default;

    Triangle(const Triangle& other) = default;
    
    constexpr Triangle(const Vec3 &a, const Vec3 &b, const Vec3 &c)
        : p0(a), p1(b), p2(c) {}
};