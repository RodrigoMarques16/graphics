#include <unit3.hpp>

#include <ostream>
#include <vec3.hpp>

constexpr Vec3 Unit3::toVec3() const { return {x, y, z}; }

constexpr bool Unit3::operator==(const Unit3 &other) const {
    return x == other.x && y == other.y && z == other.z;
}

constexpr bool Unit3::operator!=(const Unit3 &other) const {
    return x != other.x && y != other.y && z != other.z;
}

constexpr Unit3 Unit3::operator-() const {
    return {-x, -y, -z};
}

constexpr Vec3 Unit3::operator*(float k) const {
    return {x * k, y * k, z * k}; 
}

constexpr float Unit3::dot(const Vec3& other) const {
    return x * other.x + y * other.y + z * other.z;
}

constexpr float Unit3::dot(const Unit3& other) const{
    return x * other.x + y * other.y + z * other.z;
}
constexpr Vec3 Unit3::cross(const Unit3& other) const{
    return {y * other.z - z * other.y, 
            z * other.x - x * other.z,
            x * other.y - y * other.x};
}
constexpr Vec3 Unit3::cross(const Vec3& other) const{
    return {y * other.z - z * other.y, 
            z * other.x - x * other.z,
            x * other.y - y * other.x};
}

Unit3 Unit3::fromVec3(const Vec3& v) {
    return {v.x, v.y, v.z};
}

std::ostream& operator<<(std::ostream& os, const Unit3& v) {
    return os << '(' << v.x << ", " << v.y << ", " << v.z << ')';
};