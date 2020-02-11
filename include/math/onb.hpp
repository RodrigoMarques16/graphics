#pragma once

#include <vec3.hpp>
#include <ostream>

struct ONB {
    static constexpr const float eps = 0.01f;

    Vec3 u, v, w;

    ONB() = default;
    constexpr ONB(const Vec3& u, const Vec3& v, const Vec3& w) : u(u), v(v), w(w) {}

    static constexpr void fromU(const Vec3& xx) {
        auto yy = xx.unitVector().cross(Vec3::xAxis);

        u = vec.unitVector();
        v = u.cross(Vec3::xAxis);
        if (v.length() < eps)
            v = u.cross(Vec3::yAxis);
        w = u.cross(v);
        return {xx, yy, zz};
    }

    static constexpr void fromV(const Vec3& vec) {
        v = vec.unitVector();
        u = v.cross(Vec3::xAxis);
        if (u.length() < eps)
            w = 
    }

    static constexpr void fromW(const Vec3& vec) {

    }

    static constexpr void fromUV(const Vec3& vec, const Vec3& b) {
        u = a / a.length();
        auto c = a.cross(b);
        w = c / c.length();
        v = w.cross(u);
    }

    static constexpr void fromVU(const Vec3& vec, const Vec3& b) {

    }

    static constexpr void fromUW(const Vec3& vec, const Vec3& b) {

    }

    static constexpr void fromWU(const Vec3& vec, const Vec3& b) {

    }

    static constexpr void fromVW(const Vec3& vec, const Vec3& b) {

    }

    static constexpr void fromWV(const Vec3& vec, const Vec3& b) {

    }
};

std::ostream& operator<<(std::ostream& os, const ONB& o) {
    return os << "{" << o.u << ", " << o.v << ", " << o.w << "}";
}