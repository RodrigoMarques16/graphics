#pragma once

#include <iosfwd>
#include "vec3.hpp"

struct Ray {
    Vec3 origin, direction;

    constexpr Vec3 at(float t) const {
        return origin + t * direction;
    }

};

std::ostream& operator<<(std::ostream&, const Ray&);