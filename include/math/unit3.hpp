#pragma once

#include <iosfwd>

struct Vec3;

struct Norm3 {
    friend struct Vec3;

    float x, y, z;

    constexpr explicit Norm3(const Vec3&);
    constexpr Norm3(float x, float y, float z) : x(x), y(y), z(z) {}

    constexpr Vec3 toVec3() const;

    static constexpr Norm3 fromNormal(const Vec3& normal);

    constexpr bool operator==(const Norm3& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    constexpr bool operator!=(const Norm3& other) const {
        return x != other.x && y != other.y && z != other.z;
    }

    constexpr Norm3 operator+() const {
        return *this;
    }

    constexpr Norm3 operator-() const {
        return {-x, -y, -z};
    }

    constexpr Vec3 operator*(float k) const;

    constexpr float dot(const Vec3&) const;
    constexpr float dot(const Norm3&) const;
    constexpr Vec3 cross(const Vec3&) const;
    constexpr Vec3 cross(const Norm3&) const;
};

std::ostream& operator<<(std::ostream& os, const Norm3& norm);