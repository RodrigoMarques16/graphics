#pragma once

#include "../math/vec3.cuh"
#include "../math/ray.cuh"

struct Sphere {
    Vec3 center;
    float radiusSquared;

    Sphere() = default;
    Sphere(const Vec3& c, float r) : center(c), radiusSquared(r) {}

    __device__ bool hit(const Ray& r, float nearest_dist, Hit& rec) const {
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
            return true;
        }
    
        return false;
    }

};