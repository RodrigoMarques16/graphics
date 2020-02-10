#pragma once

#include <cmath>
#include <iostream>

struct Vec3 {
    static constexpr const float eps = 0.00000001;

    static const Vec3 zero;
    static const Vec3 xAxis;
    static const Vec3 yAxis;
    static const Vec3 zAxis;
    static const Vec3 up;
    static const Vec3 down;
    static const Vec3 left;
    static const Vec3 right;
    static const Vec3 forward;
    static const Vec3 back;

    float x, y, z;
    
    // constexpr Vec3() = default;
    // constexpr Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    constexpr float& operator[](int i) { return (&x)[i]; }
    constexpr const float& operator[](int i) const { return (&x)[i]; }

    constexpr Vec3 operator+() const { return *this; }
    constexpr Vec3 operator-() const { return Vec3{-x, -y, -z}; }

    constexpr Vec3 operator+(const Vec3& other) const {
        return Vec3{x + other.x, y + other.y, z + other.z};
    }

    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    constexpr Vec3 operator-(const Vec3& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    Vec3& operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    constexpr Vec3 operator*(float f) const {
        return {x * f, y * f, z * f};
    }

    constexpr friend Vec3 operator*(float f, const Vec3 v) {
        return {v.x * f, v.y * f, v.z * f};
    }

    Vec3& operator*=(float f) {
        x *= f;
        y *= f;
        z *= f;
        return *this;
    }

    constexpr Vec3 operator/(float f) {
        return {x / f, y / f, z / f};
    }

    Vec3& operator/=(float f) {
        x /= f;
        y /= f;
        z /= f;
        return *this;
    }

    constexpr Vec3 operator*(const Vec3& other) const {
        return {x * other.x, y * other.y, z * other.z};
    }

    Vec3& operator*=(const Vec3& other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return *this;
    }

    // constexpr bool operator<=>(const Vec3&) const = default;

    constexpr bool operator==(const Vec3& other) const {
        return ((*this) - other).lengthSquared() < eps;
    };

    constexpr bool operator!=(const Vec3& other) const {
        return x != other.x || y != other.y || z != other.z;
    };

    constexpr float lengthSquared() const { 
        return x * x + y * y + z * z; 
    }

    float length() const { 
        return sqrt(lengthSquared()); 
    }
    
    constexpr float dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    constexpr Vec3 cross(const Vec3& other) const {
        return {y * other.z - z * other.y, 
                z * other.x - x * other.z,
                x * other.y - y * other.x};
    }

    constexpr Vec3 normalised() const { 
        float length = this->length();
        if (length > eps)
            return (*this) * (1 / this->length());
        else return zero;
    }

};

const Vec3 Vec3::zero    = {0,  0,  0};
const Vec3 Vec3::xAxis   = {1,  0,  0};
const Vec3 Vec3::yAxis   = {0,  1,  0};
const Vec3 Vec3::zAxis   = {0,  0,  1};
const Vec3 Vec3::up      = {0,  1,  0};
const Vec3 Vec3::down    = {0, -1,  0};
const Vec3 Vec3::left    = {1,  0,  0};
const Vec3 Vec3::right   = {1,  0,  0};
const Vec3 Vec3::forward = {0,  0,  1};
const Vec3 Vec3::back    = {0,  0, -1};

std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    return os << '(' << v.x << ", " << v.y << ", " << v.z << ')';
};

void foo() {
    Vec3 vec{1,2,3};
}