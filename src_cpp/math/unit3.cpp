#include "unit3.hpp"
#include "vec3.hpp"
#include <ostream>

Unit3 Unit3::fromVec3(const Vec3& v) {
    return {v.x, v.y, v.z};
}

std::ostream& operator<<(std::ostream& os, const Unit3& u) {
    return os << '(' << u.x << ", " << u.y << ", " << u.z << ')';
};