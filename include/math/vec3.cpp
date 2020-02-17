#include "vec3.hpp"

#include <ostream>
#include <unit3.hpp>

constexpr const Vec3 Vec3::zero    = {0,  0,  0};
constexpr const Vec3 Vec3::xAxis   = {1,  0,  0};
constexpr const Vec3 Vec3::yAxis   = {0,  1,  0};
constexpr const Vec3 Vec3::zAxis   = {0,  0,  1};
constexpr const Vec3 Vec3::up      = {0,  1,  0};
constexpr const Vec3 Vec3::down    = {0, -1,  0};
constexpr const Vec3 Vec3::left    = {1,  0,  0};
constexpr const Vec3 Vec3::right   = {1,  0,  0};
constexpr const Vec3 Vec3::forward = {0,  0,  1};
constexpr const Vec3 Vec3::back    = {0,  0, -1};

constexpr Vec3::Vec3(const Unit3& u) : x(u.x), y(u.y), z(u.z) {}

constexpr float& Vec3::operator[](int i) { return (&x)[i]; }

constexpr const float& Vec3::operator[](int i) const { return (&x)[i]; }

constexpr Vec3 Vec3::operator-() const { return {-x, -y, -z}; }

constexpr Vec3 Vec3::operator+(const Vec3& other) const {
    return {x + other.x, y + other.y, z + other.z};
}

Vec3& Vec3::operator+=(const Vec3 &other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
}

constexpr Vec3 Vec3::operator-(const Vec3& other) const {
    return {x - other.x, y - other.y, z - other.z};
}

Vec3& Vec3::operator-=(const Vec3& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
}

constexpr Vec3 Vec3::operator*(float f) const {
    return {x * f, y * f, z * f};
}

constexpr Vec3 operator*(float f, const Vec3 v) {
    return {v.x * f, v.y * f, v.z * f};
}

Vec3& Vec3::operator*=(float f) {
    x *= f;
    y *= f;
    z *= f;
    return *this;
}

constexpr Vec3 Vec3::operator/(float f) const {
    return {x / f, y / f, z / f};
}

Vec3& Vec3::operator/=(float f) {
    x /= f;
    y /= f;
    z /= f;
    return *this;
}

constexpr Vec3 Vec3::operator*(const Vec3& other) const {
    return {x * other.x, y * other.y, z * other.z};
}

Vec3& Vec3::operator*=(const Vec3& other) {
    x *= other.x;
    y *= other.y;
    z *= other.z;
    return *this;
}

constexpr bool Vec3::operator==(const Vec3& other) const {
    return ((*this) - other).lengthSquared() < eps;
};

constexpr bool Vec3::operator!=(const Vec3& other) const {
    return x != other.x || y != other.y || z != other.z;
};

constexpr float Vec3::lengthSquared() const { 
     return x * x + y * y + z * z; 
}

float Vec3::length() const { 
    return sqrt(lengthSquared()); 
}

constexpr float Vec3::dot(const Vec3& other) const {
    return x * other.x + y * other.y + z * other.z;
}

constexpr Vec3 Vec3::cross(const Vec3& other) const {
    return {y * other.z - z * other.y, 
            z * other.x - x * other.z,
            x * other.y - y * other.x};
}

Unit3 Vec3::unitVector() const { 
    float length = this->length();
    if (length > eps)
        return Unit3::fromVec3((*this) * (1 / this->length()));
    else return {0,0,0};
}

std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    return os << '(' << v.x << ", " << v.y << ", " << v.z << ')';
};