#include "rgb.hpp"
#include <algorithm>
#include <iostream>

void rgb::clamp() {
    r = std::clamp(r, 0.0, 1.0);
    g = std::clamp(g, 0.0, 1.0);
    b = std::clamp(b, 0.0, 1.0);
}

std::ostream& operator<<(std::ostream& os, const rgb& c) {
    return os << "rgb(" << c.r << ", " << c.g << ", " << c.b << ')';
}