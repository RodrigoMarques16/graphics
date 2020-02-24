#include "ray.hpp"
#include <iostream>

std::ostream& operator<<(std::ostream& os, const Ray& r) {
    return os << "r{" << r.origin << " + " << "t*" << r.direction << '}';
};