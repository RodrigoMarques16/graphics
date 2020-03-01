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

    Scene() {
        // triangles = (Triangle*) malloc(100 * sizeof(Triangle));
        // cudaMalloc((void**)&triangles, 100 * sizeof(Triangle));
        // cudaMalloc((void**)&spheres, 100 * sizeof(Sphere));
    }

    __device__ void init() {
        triangles = (Triangle*) malloc(100 * sizeof(Triangle));
    }

    __device__ ~Scene() {
        free(triangles);
        // cudaFree(triangles);
        // cudaFree(spheres);
    }

    // void addTriangle(const Triangle& o) {
    //     triangles[num_triangles++] = o;
    // }

    __device__ void addTriangle(const Vec3& p0, const Vec3& p1, const Vec3& p2) {
        triangles[num_triangles++] = Triangle(p0, p1, p2);
    }

    // void addSphere(const Sphere& o) {
    //     spheres[num_spheres++] = o;
    // }

    __device__ void addSphere(const Vec3& center, float radiusSquared) {
        spheres[num_spheres++] = {center, radiusSquared};
    }

    __device__ bool intersect(const Ray& r, Hit& rec) const {
        // Hit* sphereHit   = intersectSpheres(r, max_render_dist);
        // printf("intersects\n");
        // Hit* triangleHit = intersectTriangles(r, sphereHit != nullptr ? 
        //                                             sphereHit->dist :
        //                                             max_render_dist);
        // return triangleHit ? triangleHit : sphereHit;
        return intersectTriangles(r, max_render_dist, rec);
    }

    __device__ Hit* intersectSpheres(const Ray& r, float nearest_dist) const{
        int nearest_index = -1;
    
    //     for (size_t i = 0; i < num_spheres; ++i) {
    //         Vec3 l = spheres[i].center - r.origin;
    //         float s = l.dot(r.direction);
    
    //         float lsquared = l.dot(l);
    //         if (s < 0 && lsquared > spheres[i].radiusSquared)
    //             continue;
    
    //         float msquared = lsquared - s * s;
    //         if (msquared > spheres[i].radiusSquared)
    //             continue;
    
    //         float q = sqrt(spheres[i].radiusSquared - msquared);
    //         float t;
    //         if (lsquared > spheres[i].radiusSquared)
    //             t = s - q;
    //         else t = s +q;
    
    //         if (t < nearest_dist) {
    //             nearest_index = i;
    //             nearest_dist = t;
    //         }
    //     }
    
        if (nearest_index == -1) return nullptr;
    
    //     // auto norm = (r.origin + nearestDist * r.direction - spheres[neareastIndex].center)
    //     //             .unitVector();
    
        return new Hit{nearest_dist, /*norm,*/ {.2,.2,.8}};
    }

    __device__ bool intersectTriangles(const Ray& r, float nearest_dist, Hit& rec) const {
        int nearest_index = -1;
        float nearest_u, nearest_v;
        for(int i = 0; i < num_triangles; ++i) {
            Vec3 p0p1 = triangles[i].p1 - triangles[i].p0;
            Vec3 p0p2 = triangles[i].p2 - triangles[i].p0;
            Vec3 pvec = r.direction.cross(p0p2);
            float det = p0p1.dot(pvec);
    
            if (fabs(det) < eps) continue;
    
            float invDet = 1 / det;
    
            Vec3 tvec = r.origin - triangles[i].p0;
            
            float u = tvec.dot(pvec) * invDet;
            if (u < 0 || u > 1) continue;
    
            Vec3 qvec = tvec.cross(p0p1);
            
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

        if (nearest_index == -1) return false;
    
        rec.dist = nearest_dist;
        rec.color = {nearest_u, nearest_v, 1 - nearest_u - nearest_v};

        return true;
    }
};