#pragma once

#include "hit.hpp"
#include "sphere.hpp"
#include "triangle.hpp"
#include <optional>
#include <vector>

struct Scene {
    static constexpr float max_render_dist = std::numeric_limits<float>::infinity();
    static constexpr float eps = 0.00001f;

    std::vector<Triangle> triangles;
    std::vector<Sphere> spheres;

    void addTriangle(const Triangle&);
    void addTriangle(const Vec3&, const Vec3&, const Vec3&);

    void addSphere(const Sphere&);
    void addSphere(const Vec3&, float);

    void render();

    std::optional<HitRecord> intersect(const Ray&);
    std::optional<HitRecord> intersectSpheres(const Ray&, float);
    std::optional<HitRecord> intersectTriangles(const Ray&, float);
};