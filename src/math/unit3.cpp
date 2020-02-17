#include "unit3.hpp"

#include <ostream>

std::ostream& operator<<(std::ostream& os, const Unit3& u) {
    return os << '(' << u.x << ", " << u.y << ", " << u.z << ')';
};