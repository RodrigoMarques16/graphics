#pragma once

struct rgb {
    double r, g, b;

    __host__ __device__ constexpr rgb operator-() const { return {-r, -g, -b}; }

    __host__ __device__ constexpr rgb operator+(const rgb& other) const {
        return {r + other.r, g + other.g, b + other.b};
    }

    __host__ __device__ rgb& operator+=(const rgb& other) {
        r += other.r;
        g += other.g;
        b += other.b;
        return *this;
    }

    __host__ __device__ constexpr rgb operator-(const rgb& other) const {
        return {r - other.r, g - other.g, b - other.b};
    }

    __host__ __device__ rgb& operator-=(const rgb& other) {
        r -= other.r;
        g -= other.g;
        b -= other.b;
        return *this;
    }

    __host__ __device__ constexpr rgb operator*(double f) const {
        return {r * f, g * f, b * f};
    }

    __host__ __device__ constexpr friend rgb operator*(double f, const rgb v) {
        return {v.r * f, v.g * f, v.b * f};
    }

    __host__ __device__ rgb& operator*=(double f) {
        r *= f;
        g *= f;
        b *= f;
        return *this;
    }

    __host__ __device__ constexpr rgb operator/(double f) { return {r / f, g / f, b / f}; }

    __host__ __device__ rgb& operator/=(double f) {
        r /= f;
        g /= f;
        b /= f;
        return *this;
    }

    __host__ __device__ constexpr rgb operator*(const rgb& other) const {
        return {r * other.r, g * other.g, b * other.b};
    }

    __host__ __device__ rgb& operator*=(const rgb& other) {
        r *= other.r;
        g *= other.g;
        b *= other.b;
        return *this;
    }

    __host__ __device__ void clamp() {
        r = r > 1.0 ? 1.0 : r < 0.0 ? 0.0 : r;
        g = g > 1.0 ? 1.0 : g < 0.0 ? 0.0 : g;
        b = b > 1.0 ? 1.0 : b < 0.0 ? 0.0 : b;
    }
};