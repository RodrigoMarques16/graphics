#pragma once

#include <iosfwd>

struct Vec3;

struct Unit3 {
    friend struct Vec3;

    float x, y, z;

    static Unit3 fromVec3(const Vec3&);
    
    constexpr Vec3 toVec3() const;

    constexpr bool operator==(const Unit3&) const;
    constexpr bool operator!=(const Unit3&) const;

    constexpr Unit3 operator-() const;

    constexpr Vec3 operator*(float k) const;

    constexpr float dot(const Vec3&) const;
    constexpr float dot(const Unit3&) const;
    constexpr Vec3 cross(const Vec3&) const;
    constexpr Vec3 cross(const Unit3&) const;
};

std::ostream& operator<<(std::ostream&, const Unit3&);