#pragma once

struct RGB {
    int r, g, b;

    // constexpr RGB() = default;
    // constexpr RGB(float r, float g, float b) : r(r), g(g), b(b) {};

    constexpr RGB operator+() const { return *this; }
    constexpr RGB operator-() const { return RGB(-r, -g, -b); }

    constexpr RGB operator+(const RGB& other) const {
        return {r + other.r, g + other.g, b + other.b};
    }

    RGB& operator+=(const RGB& other) {
        r += other.r;
        g += other.g;
        b += other.b;
        return *this;
    }

    constexpr RGB operator-(const RGB& other) const {
        return {r - other.r, g - other.g, b - other.b};
    }

    RGB& operator-=(const RGB& other) {
        r -= other.r;
        g -= other.g;
        b -= other.b;
        return *this;
    }

    constexpr RGB operator*(float f) const {
        return {r * f, g * f, b * f};
    }

    constexpr friend RGB operator*(float f, const RGB v) {
        return {v.r * f, v.g * f, v.b * f};
    }

    RGB& operator*=(float f) {
        r *= f;
        g *= f;
        b *= f;
        return *this;
    }

    constexpr RGB operator/(float f) { return RGB{r / f, g / f, b / f}; }

    RGB& operator/=(float f) {
        r /= f;
        g /= f;
        b /= f;
        return *this;
    }

    constexpr RGB operator*(const RGB& other) const {
        return RGB{r * other.r, g * other.g, b * other.b};
    }

    RGB& operator*=(const RGB& other) {
        r *= other.r;
        g *= other.g;
        b *= other.b;
        return *this;
    }
};