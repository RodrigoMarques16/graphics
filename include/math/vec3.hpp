#pragma once

#include <cmath>
#include <iosfwd>

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
     
    // Vec3() = default;
    // constexpr Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    constexpr float& operator[](int i);
    constexpr const float& operator[](int i) const;

    constexpr Vec3 operator-() const;

    constexpr Vec3 operator+(const Vec3& other) const;

    Vec3& operator+=(const Vec3& other);

    constexpr Vec3 operator-(const Vec3& other) const;

    Vec3& operator-=(const Vec3& other);

    constexpr Vec3 operator*(float f) const;

    constexpr friend Vec3 operator*(float f, const Vec3 v);

    Vec3& operator*=(float f);

    constexpr Vec3 operator/(float f) const;

    Vec3& operator/=(float f);

    constexpr Vec3 operator*(const Vec3& other) const;

    Vec3& operator*=(const Vec3& other);

    // constexpr bool operator<=>(const Vec3&) const = default;

    constexpr bool operator==(const Vec3& other) const;

    constexpr bool operator!=(const Vec3& other) const;

    constexpr float lengthSquared() const;
    float length() const;
    
    constexpr float dot(const Vec3& other) const;

    constexpr Vec3 cross(const Vec3& other) const;

    Vec3 unitVector() const;
};

std::ostream& operator<<(std::ostream& os, const Vec3& v);