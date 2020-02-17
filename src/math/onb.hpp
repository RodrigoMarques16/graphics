#pragma once

#include "vec3.hpp"
#include <ostream>

struct ONB {
    static constexpr const float eps = 0.00000001;

    Vec3 u, v, w;

    static constexpr void fromU(const Vec3& xx);
    static constexpr void fromV(const Vec3& vec);
    static constexpr void fromW(const Vec3& vec);
    static constexpr void fromUV(const Vec3& vec, const Vec3& b);
    static constexpr void fromVU(const Vec3& vec, const Vec3& b);
    static constexpr void fromUW(const Vec3& vec, const Vec3& b);
    static constexpr void fromWU(const Vec3& vec, const Vec3& b);
    static constexpr void fromVW(const Vec3& vec, const Vec3& b);
    static constexpr void fromWV(const Vec3& vec, const Vec3& b);
};

std::ostream& operator<<(std::ostream& os, const ONB& o);