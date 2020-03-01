#pragma once

#include "../math/ray.cuh"
#include "sphere.cuh"
#include "triangle.cuh"

struct Scene {
    static constexpr const float max_render_dist = std::numeric_limits<float>::infinity();
    static constexpr const float eps = 0.000001;

    int num_triangles = 0;
    Triangle* triangles;
    
    long long num_spheres = 0;
    Sphere* spheres;

    Scene() = default;

    __device__ void init() {
        triangles = (Triangle*) malloc(100 * sizeof(Triangle));
        spheres = (Sphere*) malloc(100 * sizeof(Sphere));
    }

    __device__ ~Scene() {
        printf("goodbye!\n");
        free(triangles);
        free(spheres);
    }

    __device__ void addTriangle(const Vec3& p0, const Vec3& p1, const Vec3& p2) {
        triangles[num_triangles++] = Triangle(p0, p1, p2);
    }

    __device__ void addSphere(const Vec3& center, float radiusSquared) {
        spheres[num_spheres++] = {center, radiusSquared};
    }

    __device__ bool intersect(const Ray& r, Hit& rec) const {
        bool hitSphere = intersectSpheres(r, max_render_dist, rec);
        return hitSphere | intersectTriangles(r, hitSphere ? rec.dist : max_render_dist, rec);
    }

    __device__ bool intersectSpheres(const Ray& r, float nearest_dist, Hit& rec) const {
        bool gotHit = false;
        for (size_t i = 0; i < num_spheres; ++i)
            gotHit = gotHit | spheres[i].hit(r, nearest_dist, rec);
        return gotHit;
    }

    __device__ bool intersectTriangles(const Ray& r, float nearest_dist, Hit& rec) const {
        bool gotHit = false;
        for (size_t i = 0; i < num_triangles; ++i)
            gotHit = gotHit | triangles[i].hit(r, nearest_dist, rec);
        return gotHit;
    }
};