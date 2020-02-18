#include "rgb.hpp"
#include <algorithm>
#include <iostream>

void RGB::clamp() {
    r = std::clamp(r, 0.0f, 1.0f);
    g = std::clamp(g, 0.0f, 1.0f);
    b = std::clamp(b, 0.0f, 1.0f);
}

std::ostream& operator<<(std::ostream& os, const RGB& c) {
    return os << "rgb(" << c.r << ", " << c.g << ", " << c.b << ')';
}