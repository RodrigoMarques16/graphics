#include "onb.hpp"

#include <ostream>

std::ostream& operator<<(std::ostream& os, const ONB& o) {
    return os << "{" << o.u << ", " << o.v << ", " << o.w << "}";
}