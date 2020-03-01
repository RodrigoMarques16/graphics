#pragma once

#include <cmath>

struct Vec3 {
    static constexpr const float eps = 0.000001f;

    float x, y, z;

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

    __host__ __device__ constexpr Vec3 operator*(float f) const { return {x * f, y * f, z * f}; }

    __host__ __device__ constexpr friend Vec3 operator*(float f, const Vec3 v) {
        return {v.x * f, v.y * f, v.z * f};
    }

    __host__ __device__ Vec3& operator*=(float f) {
        x *= f;
        y *= f;
        z *= f;
        return *this;
    }

    __host__ __device__ constexpr Vec3 operator/(float f) const { return {x / f, y / f, z / f}; }

    __host__ __device__ Vec3& operator/=(float f) {
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

    __host__ __device__ constexpr float lengthSquared() const { 
        return x * x + y * y + z * z; 
    }
    
    __host__ __device__ float length() const { 
        return sqrt(lengthSquared()); 
    }

    __host__ __device__ constexpr float dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    __host__ __device__ constexpr Vec3 cross(const Vec3& other) const {
        return {y * other.z - z * other.y, 
                z * other.x - x * other.z,
                x * other.y - y * other.x};
    }

    __host__ __device__ Vec3 unitVector() const {
        float length = this->length();
        if (length > eps)
            return (*this) * (1 / this->length());
        else return {0,0,0};
    }
};