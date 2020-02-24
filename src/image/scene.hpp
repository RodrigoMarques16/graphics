#pragma once

#include "hit.hpp"
#include "sphere.hpp"
#include "triangle.hpp"
#include <optional>
#include <vector>

struct Scene {
    std::vector<Triangle> triangles;
    std::vector<Sphere> spheres;

    void addTriangle(const Triangle&);
    void addTriangle(const Vec3&, const Vec3&, const Vec3&, const rgb&);

    void addSphere(const Triangle&);
    void addSphere(const Vec3&, float);

    void render();

    std::optional<HitRecord> intersect(const Ray&);
    std::optional<HitRecord> intersectSpheres(const Ray&);
    std::optional<HitRecord> intersectTriangles(const Ray&);
};