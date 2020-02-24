#pragma once

#include "hit.hpp"
#include "ray.hpp"
#include "vec3.hpp"
#include <optional>

struct Triangle {
    Vec3 p0, p1, p2;
    rgb color;

    constexpr Triangle(const Vec3 &a, const Vec3 &b, const Vec3 &c,
                       const rgb &col)
        : p0(a), p1(b), p2(c), color(col) {}

    constexpr Triangle(const Triangle &other) = default;

    constexpr std::optional<HitRecord> hit(const Ray& r, float tmin, float tmax) const {
        float A = p0.x - p1.x;
        float B = p0.y - p1.y;
        float C = p0.z - p1.z;

        float D = p0.x - p2.x;
        float E = p0.y - p2.y;
        float F = p0.z - p2.z;

        float G = r.direction.x;
        float H = r.direction.y;
        float I = r.direction.z;

        float J = p0.x - r.origin.x;
        float K = p0.y - r.origin.y;
        float L = p0.z - r.origin.z;

        float EIHF = E*I - H*F;
        float GFDI = G*F - D*I;
        float DHEG = D*H - E*G;

        float denom = (A*EIHF + B*GFDI + C*DHEG);
        float beta = (J*EIHF + K*GFDI + L*DHEG) / denom; 

        if (beta <= 0.0f || beta >= 1.0f) return std::nullopt;

        float AKJB = A*K - J*B;
        float JCAL = J*C - A*L;
        float BLKC = B*L - K*C;

        float gamma = (I*AKJB + H*JCAL + G*BLKC) / denom;

        if (gamma <= 0.0f || beta + gamma >= 1.0f) return std::nullopt;

        float tval = -(F*AKJB + E*JCAL + D*BLKC) / denom;
        if (tval >= tmin && tval <= tmax) {
            auto norm = (p1-p0).cross(p2-p0).unitVector();
            return std::make_optional<HitRecord>({tval, norm, color});
        }

        return std::nullopt;
    }

    constexpr bool shadowHit(const Ray& r, float tmin, float tmax) const {
        float A = p0.x - p1.x;
        float B = p0.y - p1.y;
        float C = p0.z - p1.z;

        float D = p0.x - p2.x;
        float E = p0.y - p2.y;
        float F = p0.z - p2.z;

        float G = r.direction.x;
        float H = r.direction.y;
        float I = r.direction.z;

        float J = p0.x - r.origin.x;
        float K = p0.y - r.origin.y;
        float L = p0.z - r.origin.z;

        float EIHF = E*I - H*F;
        float GFDI = G*F - D*I;
        float DHEG = D*H - E*G;

        float denom = (A*EIHF + B*GFDI + C*DHEG);
        float beta = (J*EIHF + K*GFDI + L*DHEG) / denom; 

        if (beta <= 0.0f || beta >= 1.0f) return false;

        float AKJB = A*K - J*B;
        float JCAL = J*C - A*L;
        float BLKC = B*L - K*C;

        float gamma = (I*AKJB + H*JCAL + G*BLKC) / denom;

        if (gamma <= 0.0f || gamma >= 1.0f) return false;

        float tval = -(F*AKJB + E*JCAL + D*BLKC) / denom;
        
        return tval >= tmin 
            && tval <= tmax;
    }
};