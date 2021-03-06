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

void Scene::addSphere(const Vec3& center, double radiusSquared) {
    spheres.emplace_back(center, radiusSquared);
}

std::optional<HitRecord> Scene::intersect(const Ray& r) {
    auto sphereHit   = intersectSpheres(r, max_render_dist);
    auto triangleHit = intersectTriangles(r, sphereHit ? sphereHit->dist 
                                                       : max_render_dist);
    return triangleHit ? triangleHit : sphereHit;
}

std::optional<std::pair<double, double>> solveQuadratic(double a, double b, double c) {
    auto discriminant = b * b - 4 * a * c;
    double t0, t1;
    if (discriminant < 0) return std::nullopt;
    if (discriminant == 0) {
        t0 = t1 = -0.5 * b / a;
    } else {
        double q = (b > 0) ? -0.5 * (b + sqrt(discriminant))
                          : -0.5 * (b - sqrt(discriminant));
        t0 = q / a;
        t1 = c / q;
    }
    return std::make_pair(t0, t1);
}

// real-time rendering 4th edition
std::optional<HitRecord> Scene::intersectSpheres(const Ray& r, double nearest_dist) {
    int nearest_index = -1;

    for (size_t i = 0; i < spheres.size(); ++i) {
        auto l = spheres[i].center - r.origin;
        auto s = l.dot(r.direction);

        auto lsquared = l.dot(l);
        if (s < 0 && lsquared > spheres[i].radiusSquared)
            continue;

        auto msquared = lsquared - s * s;
        if (msquared > spheres[i].radiusSquared)
            continue;

        auto q = sqrt(spheres[i].radiusSquared - msquared);
        float t;
        if (lsquared > spheres[i].radiusSquared)
            t = s - q;
        else t = s +q;

        if (t < nearest_dist) {
            nearest_index = i;
            nearest_dist = t;
        }

        // auto temp = r.origin - spheres[i].center;
        // auto a = r.direction.lengthSquared();
        // auto b = 2 * r.direction.dot(temp);
        // auto c = temp.lengthSquared() - spheres[i].radiusSquared;

        // auto sol = solveQuadratic(a, b, c);
        
        // if (!sol.has_value()) continue;

        // auto [t0, t1] = sol.value();

        // if (t0 > t1) std::swap(t0,t1);

        // if (t0 < 0) {
        //     t0 = t1;
        //     if (t0 < 0) continue;
        // }
        // if (t0 < nearest_dist) {
        //     nearest_index = i;
        //     nearest_dist = t0;
        // }
    }

    if (nearest_index == -1) return std::nullopt;

    // auto norm = (r.origin + nearestDist * r.direction - spheres[neareastIndex].center)
    //             .unitVector();

    return HitRecord{nearest_dist, /*norm,*/ {.2,.2,.8}};
}


// https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
std::optional<HitRecord> Scene::intersectTriangles(const Ray& r, double nearest_dist) {
    int nearest_index = -1;
    double nearest_u, nearest_v;

    for(size_t i = 0; i < triangles.size(); ++i) {
        auto p0p1 = triangles[i].p1 - triangles[i].p0;
        auto p0p2 = triangles[i].p2 - triangles[i].p0;
        auto pvec = r.direction.cross(p0p2);
        double det = p0p1.dot(pvec);

        if (fabs(det) < eps) continue;

        double invDet = 1 / det;

        auto tvec = r.origin - triangles[i].p0;
        
        double u = tvec.dot(pvec) * invDet;
        if (u < 0 || u > 1) continue;

        auto qvec = tvec.cross(p0p1);
        
        double v = r.direction.dot(qvec) * invDet;
        if (v < 0 || u + v > 1) continue;

        double t = p0p2.dot(qvec) * invDet;

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