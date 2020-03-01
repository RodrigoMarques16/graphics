#pragma once

#include "../math/vec3.cuh"
#include "../math/ray.cuh"

struct Sphere {
    static constexpr const float eps = 0.000001;
    Vec3 center;
    float radiusSquared;

    __device__ Sphere() {};
    __device__ Sphere(const Vec3& c, float r) : center(c), radiusSquared(r) {}

    __device__ 
    bool hit(const Ray& r, float nearest_dist, Hit& rec) const {
        Vec3 l = center - r.origin;
        float s = l.dot(r.direction);
    
        float lsquared = l.dot(l);
        if (s < 0 && lsquared > radiusSquared)
            return false;
    
        float msquared = lsquared - s * s;
        if (msquared > radiusSquared)
            return false;
    
        float q = sqrt(radiusSquared - msquared);
        
        float t;
        if (lsquared > radiusSquared)
            t = s - q;
        else t = s + q;
    
        if (t < nearest_dist) {
            rec.dist = t;
            rec.color = {.2,.2,.8};
            return true;
        }
    
        return false;
    }

};