#pragma once

#include <cmath>

struct Vec3 {
    static constexpr const double eps = 0.000001;

    double x, y, z;

    __host__ __device__ constexpr Vec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ constexpr Vec3 operator+(const Vec3& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    __host__ __device__ Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    __host__ __device__ constexpr Vec3 operator-(const Vec3& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    __host__ __device__ Vec3& operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    __host__ __device__ constexpr Vec3 operator*(double f) const { return {x * f, y * f, z * f}; }

    __host__ __device__ constexpr friend Vec3 operator*(double f, const Vec3 v) {
        return {v.x * f, v.y * f, v.z * f};
    }

    __host__ __device__ Vec3& operator*=(double f) {
        x *= f;
        y *= f;
        z *= f;
        return *this;
    }

    __host__ __device__ constexpr Vec3 operator/(double f) const { return {x / f, y / f, z / f}; }

    __host__ __device__ Vec3& operator/=(double f) {
        x /= f;
        y /= f;
        z /= f;
        return *this;
    }

    __host__ __device__ constexpr Vec3 operator*(const Vec3& other) const {
        return {x * other.x, y * other.y, z * other.z};
    }

    __host__ __device__ Vec3& operator*=(const Vec3& other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return *this;
    }

    __host__ __device__ constexpr bool operator==(const Vec3& other) const {
        return ((*this) - other).lengthSquared() < eps;
    };

    __host__ __device__ constexpr bool operator!=(const Vec3& other) const {
        return x != other.x || y != other.y || z != other.z;
    };

    __host__ __device__ constexpr double lengthSquared() const { 
        return x * x + y * y + z * z; 
    }
    
    __host__ __device__ double length() const { 
        return sqrt(lengthSquared()); 
    }

    __host__ __device__ constexpr double dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    __host__ __device__ constexpr Vec3 cross(const Vec3& other) const {
        return {y * other.z - z * other.y, 
                z * other.x - x * other.z,
                x * other.y - y * other.x};
    }

    __host__ __device__ Vec3 unitVector() const {
        double length = this->length();
        if (length > eps)
            return (*this) * (1 / this->length());
        else return {0,0,0};
    }
};