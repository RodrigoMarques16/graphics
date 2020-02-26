#pragma once

#include <iosfwd>

struct rgb {
    double r, g, b;

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

    constexpr rgb operator*(double f) const {
        return {r * f, g * f, b * f};
    }

    constexpr friend rgb operator*(double f, const rgb v) {
        return {v.r * f, v.g * f, v.b * f};
    }

    rgb& operator*=(double f) {
        r *= f;
        g *= f;
        b *= f;
        return *this;
    }

    constexpr rgb operator/(double f) { return {r / f, g / f, b / f}; }

    rgb& operator/=(double f) {
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