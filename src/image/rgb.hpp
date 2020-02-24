#pragma once

#include <iosfwd>

struct rgb {
    float r, g, b;

    constexpr rgb operator-() const { return {-r, -g, -b}; }

    constexpr rgb operator+(const rgb& other) const {
        return {r + other.r, g + other.g, b + other.b};
    }

    rgb& operator+=(const rgb& other) {
        r += other.r;
        g += other.g;
        b += other.b;
        return *this;
    }

    constexpr rgb operator-(const rgb& other) const {
        return {r - other.r, g - other.g, b - other.b};
    }

    rgb& operator-=(const rgb& other) {
        r -= other.r;
        g -= other.g;
        b -= other.b;
        return *this;
    }

    constexpr rgb operator*(float f) const {
        return {r * f, g * f, b * f};
    }

    constexpr friend rgb operator*(float f, const rgb v) {
        return {v.r * f, v.g * f, v.b * f};
    }

    rgb& operator*=(float f) {
        r *= f;
        g *= f;
        b *= f;
        return *this;
    }

    constexpr rgb operator/(float f) { return {r / f, g / f, b / f}; }

    rgb& operator/=(float f) {
        r /= f;
        g /= f;
        b /= f;
        return *this;
    }

    constexpr rgb operator*(const rgb& other) const {
        return {r * other.r, g * other.g, b * other.b};
    }

    rgb& operator*=(const rgb& other) {
        r *= other.r;
        g *= other.g;
        b *= other.b;
        return *this;
    }

    void clamp();
};

std::ostream& operator<<(std::ostream&, const rgb&);