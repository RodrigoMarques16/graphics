#pragma once

#include <iosfwd>

struct Vec3;

struct Unit3 {
    float x, y, z;

    static const Unit3 zero;
    static const Unit3 xAxis;
    static const Unit3 yAxis;
    static const Unit3 zAxis;
    static const Unit3 up;
    static const Unit3 down;
    static const Unit3 left;
    static const Unit3 right;
    static const Unit3 forward;
    static const Unit3 back;

    constexpr bool operator==(const Unit3& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    constexpr bool operator!=(const Unit3& other) const {
        return x != other.x && y != other.y && z != other.z;
    }

    constexpr Unit3 operator-() const {
        return {-x, -y, -z};
    }
    
    constexpr float dot(const Unit3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    static Unit3 fromVec3(const Vec3&);
    constexpr Vec3 toVec3() const;
    constexpr Vec3 operator*(float) const;
    constexpr float dot(const Vec3&) const;
    constexpr Vec3 cross(const Vec3&) const;
    constexpr Vec3 cross(const Unit3&) const;
};

constexpr const Unit3 Unit3::zero    = {0,  0,  0};
constexpr const Unit3 Unit3::xAxis   = {1,  0,  0};
constexpr const Unit3 Unit3::yAxis   = {0,  1,  0};
constexpr const Unit3 Unit3::zAxis   = {0,  0,  1};
constexpr const Unit3 Unit3::up      = {0,  1,  0};
constexpr const Unit3 Unit3::down    = {0, -1,  0};
constexpr const Unit3 Unit3::left    = {-1,  0, 0};
constexpr const Unit3 Unit3::right   = {1,  0,  0};
constexpr const Unit3 Unit3::forward = {0,  0,  1};
constexpr const Unit3 Unit3::back    = {0,  0, -1};

#include "vec3.hpp"

constexpr Vec3 Unit3::toVec3() const {
    return {x, y, z};
}

constexpr Vec3 Unit3::operator*(float k) const {
    return {x * k, y* k, z* k};
};

constexpr float Unit3::dot(const Vec3& other) const {
    return x * other.x + y * other.y + z * other.z;
}

constexpr Vec3 Unit3::cross(const Vec3& other) const {
    return {y * other.z - z * other.y, 
            z * other.x - x * other.z,
            x * other.y - y * other.x};
}

constexpr Vec3 Unit3::cross(const Unit3& other) const {
    return {y * other.z - z * other.y, 
            z * other.x - x * other.z,
            x * other.y - y * other.x};
}

std::ostream& operator<<(std::ostream& os, const Unit3& norm);