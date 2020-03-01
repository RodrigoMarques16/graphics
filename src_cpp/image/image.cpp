#include "image.hpp"
#include <iostream>
#include <fstream>

void Image::writePPM(std::ostream& os) {
    os << "P6\n" << width << ' ' << height << '\n' << "255\n";
    for (int i = height - 1; i >= 0; --i) {
        for (int j = 0; j < width; ++j) {
            auto color  = get(j, i);
            uint ired   = std::min(255u, (unsigned int) (256u * color.r));
            uint igreen = std::min(255u, (unsigned int) (256u * color.g));
            uint iblue  = std::min(255u, (unsigned int) (256u * color.b));
            os << (unsigned char) ired
               << (unsigned char) igreen
               << (unsigned char) iblue;
        }
    }
}
