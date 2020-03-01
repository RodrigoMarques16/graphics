#pragma once

#include <cmath>
#include <iosfwd>

struct Unit3;

struct Vec2 {
    static constexpr const double eps = 0.000001;

    double x, y;
    
    static const Vec2 zero;
    static const Vec2 xAxis;
    static const Vec2 yAxis;
    static const Vec2 up;
    static const Vec2 down;
    static const Vec2 left;
    static const Vec2 right;

    constexpr double& operator[](int i) { return (&x)[i]; }
    constexpr const double& operator[](int i) const { return (&x)[i]; }

    constexpr Vec2 operator-() const { return {-x, -y}; }

    constexpr Vec2 operator+(const Vec2& other) const {
        return {x + other.x, y + other.y};
    }

    Vec2& operator+=(const Vec2& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    constexpr Vec2 operator-(const Vec2& other) const {
        return {x - other.x, y - other.y};
    }

    Vec2& operator-=(const Vec2& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    constexpr Vec2 operator*(double f) const { return {x * f, y * f}; }

    constexpr friend Vec2 operator*(double f, const Vec2 v) {
        return {v.x * f, v.y * f};
    }

    Vec2& operator*=(double f) {
        x *= f;
        y *= f;
        return *this;
    }

    constexpr Vec2 operator/(double f) const { return {x / f, y / f}; }

    Vec2& operator/=(double f) {
        x /= f;
        y /= f;
        return *this;
    }

    constexpr Vec2 operator*(const Vec2& other) const {
        return {x * other.x, y * other.y};
    }

    Vec2& operator*=(const Vec2& other) {
        x *= other.x;
        y *= other.y;
        return *this;
    }

    // constexpr bool operator<=>(const Vec2&) const = default;

    constexpr bool operator==(const Vec2& other) const {
        return ((*this) - other).lengthSquared() < eps;
    };

    constexpr bool operator!=(const Vec2& other) const {
        return x != other.x || y != other.y;
    };

    constexpr double lengthSquared() const { 
        return x * x + y * y; 
    }
    
    double length() const { 
        return sqrtf(lengthSquared()); 
    }

    constexpr double dot(const Vec2& other) const {
        return x * other.x + y * other.y;
    }
};

constexpr const Vec2 Vec2::zero  = {0,  0};
constexpr const Vec2 Vec2::xAxis = {1,  0};
constexpr const Vec2 Vec2::yAxis = {0,  1};
constexpr const Vec2 Vec2::up    = {0,  1};
constexpr const Vec2 Vec2::down  = {0, -1};
constexpr const Vec2 Vec2::left  = {-1, 0};
constexpr const Vec2 Vec2::right = {1,  0};