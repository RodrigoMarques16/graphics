#pragma once

#include "../math/vec3.cuh"

struct Triangle {
    static constexpr const float eps = 0.000001;
    Vec3 p0, p1, p2;

    __device__ Triangle() {};
    
    __device__ Triangle(const Vec3 &a, const Vec3 &b, const Vec3 &c)
        : p0(a), p1(b), p2(c) {}

    __device__ 
    bool hit(const Ray& r, float nearest_dist, Hit& rec) const {
        Vec3 p0p1 = p1 - p0;
        Vec3 p0p2 = p2 - p0;
        Vec3 pvec = r.direction.cross(p0p2);
        float det = p0p1.dot(pvec);

        if (fabs(det) < eps)
            return false;

        float invDet = 1 / det;

        Vec3 tvec = r.origin - p0;
        
        float u = tvec.dot(pvec) * invDet;
        if (u < 0 || u > 1)
            return false;

        Vec3 qvec = tvec.cross(p0p1);
        
        float v = r.direction.dot(qvec) * invDet;
        if (v < 0 || u + v > 1) 
            return false;;

        float t = p0p2.dot(qvec) * invDet;

        if (t < eps || t > nearest_dist) 
            return false;
                
        if (t < nearest_dist) {
            rec.dist = t;
            rec.color = {u, v, 1 - u - v};
            return true;
        }
        
        return false;
    }
};