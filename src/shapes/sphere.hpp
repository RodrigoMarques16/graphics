#pragma once

#include "hit.hpp"
#include "rgb.hpp"
#include "ray.hpp"
#include "vec3.hpp"
#include <optional>

struct Sphere {
    Vec3 center;
    float radiusSquared;

    constexpr Sphere(const Vec3& c, float r) : center(c), radiusSquared(r) {}

    // constexpr std::optional<HitRecord> hit(const Ray& r, float tmin, float tmax) {
    //     Vec3 temp = r.origin - center;
    //     auto a = r.direction.dot(r.direction);
    //     auto b = 2 * r.direction.dot(temp);
    //     auto c = temp.dot(temp) - radius * radius;

    //     auto discriminant = b * b - 4 * a * c;

    //     if (discriminant > 0) {
    //         discriminant = sqrt(discriminant);
    //         auto t = (-b - discriminant) / (2 * a);
    //         if (t < tmin)
    //             t = (-b + discriminant) / (2 * a);
    //         if (t < tmin || t > tmax)
    //             return std::nullopt;
    //         auto norm = (r.origin + t * r.direction - center).unitVector();
    //         return std::make_optional<HitRecord>({t, /*norm,*/ color});
    //     }

    //     return std::nullopt;
    // }

    // constexpr bool shadowHit(const Ray& r, float tmin, float tmax) {
    //     Vec3 temp = r.origin - center;
    //     auto a = r.direction.dot(r.direction);
    //     auto b = 2 * r.direction.dot(temp);
    //     auto c = temp.dot(temp) - radius * radius;

    //     auto discriminant = b * b - 4 * a * c;

    //     if (discriminant > 0) {
    //         discriminant = sqrt(discriminant);
    //         auto t = (-b - discriminant) / (2 * a);
    //         if (t < tmin)
    //             t = (-b + discriminant) / (2 * a);
    //         if (t < tmin || t > tmax)
    //             return false;
    //         return true;
    //     }

    //     return false;
    // }
};