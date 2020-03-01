#pragma once

#include "hit.hpp"
#include "sphere.hpp"
#include "triangle.hpp"
#include <optional>
#include <vector>

struct Scene {
    static constexpr const double max_render_dist = std::numeric_limits<double>::infinity();
    static constexpr const double eps = 0.000001;

    std::vector<Triangle> triangles;
    std::vector<Sphere> spheres;

    void addTriangle(const Triangle&);
    void addTriangle(const Vec3&, const Vec3&, const Vec3&);

    void addSphere(const Sphere&);
    void addSphere(const Vec3&, double);

    void render();

    std::optional<HitRecord> intersect(const Ray&);
    std::optional<HitRecord> intersectSpheres(const Ray&, double);
    std::optional<HitRecord> intersectTriangles(const Ray&, double);
};