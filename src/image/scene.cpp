#include "scene.hpp"
#include <iostream>

void Scene::addTriangle(const Triangle& o) {
    triangles.emplace_back(o);
}

void Scene::addTriangle(const Vec3& p0, const Vec3& p1, const Vec3& p2) {
    triangles.emplace_back(p0, p1, p2);
}

void Scene::addSphere(const Sphere& o) {
    spheres.emplace_back(o);
}

void Scene::addSphere(const Vec3& center, float radiusSquared) {
    spheres.emplace_back(center, radiusSquared);
}

std::optional<HitRecord> Scene::intersect(const Ray& r) {
    auto sphereHit   = intersectSpheres(r, max_render_dist);
    auto triangleHit = intersectTriangles(r, sphereHit ? sphereHit->dist 
                                                       : max_render_dist);
    return triangleHit ? triangleHit : sphereHit;
}

std::optional<std::pair<float, float>> solveQuadratic(float a, float b, float c) {
    auto discriminant = b * b - 4 * a * c;
    float t0, t1;
    if (discriminant < 0) return std::nullopt;
    if (discriminant == 0) {
        t0 = t1 = -0.5f * b / a;
    } else {
        float q = (b > 0) ? -0.5f * (b + sqrt(discriminant))
                          : -0.5f * (b - sqrt(discriminant));
        t0 = q / a;
        t1 = c / q;
    }
    return std::make_pair(t0, t1);
}

std::optional<HitRecord> Scene::intersectSpheres(const Ray& r, float nearest_dist) {
    int nearest_index = -1;

    for (size_t i = 0; i < spheres.size(); ++i) {
        auto temp = r.origin - spheres[i].center;
        auto a = r.direction.lengthSquared();
        auto b = 2 * r.direction.dot(temp);
        auto c = temp.lengthSquared() - spheres[i].radiusSquared;

        auto sol = solveQuadratic(a, b, c);
        
        if (!sol.has_value()) continue;

        auto [t0, t1] = sol.value();

        if (t0 > t1) std::swap(t0,t1);

        if (t0 < 0) {
            t0 = t1;
            if (t0 < 0) continue;
        }
        if (t0 < nearest_dist) {
            nearest_index = i;
            nearest_dist = t0;
        }
    }

    if (nearest_index == -1) return std::nullopt;

    // auto norm = (r.origin + nearestDist * r.direction - spheres[neareastIndex].center)
    //             .unitVector();

    return HitRecord{nearest_dist, /*norm,*/ {.2,.2,.8}};
}

std::optional<HitRecord> Scene::intersectTriangles(const Ray& r, float nearest_dist) {
    int nearest_index = -1;

    float nearest_u, nearest_v;

    for(size_t i = 0; i < triangles.size(); ++i) {
        auto p0p1 = triangles[i].p1 - triangles[i].p0;
        auto p0p2 = triangles[i].p2 - triangles[i].p0;
        auto pvec = r.direction.cross(p0p2);
        float det = p0p1.dot(pvec);

        if (fabs(det) < eps) continue;

        float invDet = 1 / det;
        auto tvec = r.origin - triangles[i].p0;
        float u = tvec.dot(pvec) * invDet;

        if (u < 0 || u > 1) continue;

        auto qvec = tvec.cross(p0p1);
        
        float v = r.direction.dot(qvec) * invDet;

        if (v < 0 || u + v > 1) continue;

        float t = p0p2.dot(qvec) * invDet;

        if (t < eps || t > nearest_dist) continue;

        if (t < nearest_dist) {
            nearest_index = i;
            nearest_dist = t;
            nearest_u = u;
            nearest_v = v;
        }
    }

    if (nearest_index == -1) return std::nullopt;

    return HitRecord{nearest_dist,
                     {nearest_u, nearest_v, 1 - nearest_u - nearest_v}};
}