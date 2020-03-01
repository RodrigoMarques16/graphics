#pragma once

#include "../math/ray.cuh"
#include "sphere.cuh"
#include "triangle.cuh"

struct Scene {
    static constexpr const double max_render_dist = std::numeric_limits<double>::infinity();
    static constexpr const double eps = 0.000001;

    long long num_triangles = 0;
    Triangle** triangles;
    
    long long num_spheres = 0;
    Sphere** spheres;

    Scene() {
        cudaMallocManaged((void**)&triangles, 100 * sizeof(Triangle*));
        cudaMallocManaged((void**)&spheres, 100 * sizeof(Sphere*));
    }

    ~Scene() {
        cudaFree(triangles);
        cudaFree(spheres);
    }

    // void addTriangle(const Triangle& o) {
    //     triangles[num_triangles++] = o;
    // }

    void addTriangle(const Vec3& p0, const Vec3& p1, const Vec3& p2) {
        triangles[num_triangles++] = new Triangle(p0, p1, p2);
    }

    // void addSphere(const Sphere& o) {
    //     spheres[num_spheres++] = o;
    // }

    // void addSphere(const Vec3& center, double radiusSquared) {
    //     spheres[num_spheres++] = {center, radiusSquared};
    // }

    __device__ Hit* intersect(const Ray& r) const {
        Hit* sphereHit   = intersectSpheres(r, max_render_dist);
        Hit* triangleHit = intersectTriangles(r, sphereHit ? sphereHit->dist 
                                                           : max_render_dist);
        return triangleHit ? triangleHit : sphereHit;
    }

    __device__ Hit* intersectSpheres(const Ray& r, double nearest_dist) const{
        int nearest_index = -1;
    
    //     for (size_t i = 0; i < num_spheres; ++i) {
    //         Vec3 l = spheres[i].center - r.origin;
    //         double s = l.dot(r.direction);
    
    //         double lsquared = l.dot(l);
    //         if (s < 0 && lsquared > spheres[i].radiusSquared)
    //             continue;
    
    //         double msquared = lsquared - s * s;
    //         if (msquared > spheres[i].radiusSquared)
    //             continue;
    
    //         double q = sqrt(spheres[i].radiusSquared - msquared);
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
    
    //     return new Hit{nearest_dist, /*norm,*/ {.2,.2,.8}};
    }

    __device__ Hit* intersectTriangles(const Ray& r, double nearest_dist) const{
        int nearest_index = -1;
        double nearest_u, nearest_v;
    
        for(size_t i = 0; i < num_triangles; ++i) {
            Vec3 p0p1 = triangles[i]->p1 - triangles[i]->p0;
            Vec3 p0p2 = triangles[i]->p2 - triangles[i]->p0;
            Vec3 pvec = r.direction.cross(p0p2);
            double det = p0p1.dot(pvec);
    
            if (fabs(det) < eps) continue;
    
            double invDet = 1 / det;
    
            Vec3 tvec = r.origin - triangles[i]->p0;
            
            double u = tvec.dot(pvec) * invDet;
            if (u < 0 || u > 1) continue;
    
            Vec3 qvec = tvec.cross(p0p1);
            
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
    
        if (nearest_index == -1) return nullptr;
    
        return new Hit{nearest_dist,
                         {nearest_u, nearest_v, 1 - nearest_u - nearest_v}};
    }
};