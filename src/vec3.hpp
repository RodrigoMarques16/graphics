#pragma once

#include <cmath>
#include <iostream>

struct Vec3;

struct Vec3 {
    static constexpr const double eps = 0.00000001;

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

    double x, y, z;
    
    Vec3() = default;
    constexpr Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    constexpr double& operator[](int i) { return (&x)[i]; }
    constexpr const double& operator[](int i) const { return (&x)[i]; }

    constexpr Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    constexpr Vec3 operator+() const { return *this; }
    constexpr Vec3 operator-() const { return Vec3(-x, -y, -z); }

    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    constexpr Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    Vec3& operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    constexpr Vec3 operator*(double f) const {
        return Vec3(x * f, y * f, z * f);
    }

    constexpr friend Vec3 operator*(double f, const Vec3 v) {
        return Vec3(v.x * f, v.y * f, v.z * f);
    }

    Vec3& operator*=(double f) {
        x *= f;
        y *= f;
        z *= f;
        return *this;
    }

    constexpr Vec3 operator/(double f) {
        return Vec3(x / f, y / f, z / f);
    }

    Vec3& operator/=(double f) {
        x /= f;
        y /= f;
        z /= f;
        return *this;
    }

    constexpr Vec3 operator*(const Vec3& other) const {
        return Vec3(x * other.x, y * other.y, z * other.z);
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

    constexpr double lengthSquared() const { 
        return x * x + y * y + z * z; 
    }

    double length() const { 
        return sqrt(lengthSquared()); 
    }
    
    constexpr double dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    constexpr Vec3 cross(const Vec3& other) const {
        return Vec3(y * other.z - z * other.y, 
                    z * other.x - x * other.z,
                    x * other.y - y * other.x);
    }

    constexpr Vec3 normalised() const { 
        double length = this->length();
        if (length > eps)
            return (*this) * (1 / this->length());
        else return zero;
    }

};

const Vec3 Vec3::zero    = Vec3( 0,  0,  0);
const Vec3 Vec3::xAxis   = Vec3( 1,  0,  0);
const Vec3 Vec3::yAxis   = Vec3( 0,  1,  0);
const Vec3 Vec3::zAxis   = Vec3( 0,  0,  1);
const Vec3 Vec3::up      = Vec3( 0,  1,  0);
const Vec3 Vec3::down    = Vec3( 0, -1,  0);
const Vec3 Vec3::left    = Vec3(-1,  0,  0);
const Vec3 Vec3::right   = Vec3( 1,  0,  0);
const Vec3 Vec3::forward = Vec3( 0,  0,  1);
const Vec3 Vec3::back    = Vec3( 0,  0, -1);

std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    return os << '(' << v.x << ", " << v.y << ", " << v.z << ')';
};