#pragma once

#include "rgb.hpp"
#include <algorithm>
#include <cmath>
#include <iosfwd>
#include <string>
#include <vector>

struct Image {
    int width, height;
    std::vector<rgb> raster;

    Image() = default;
    Image(int w, int h) : width(w), height(h) {
        raster.resize(w * h);
    }

    Image(int w, int h, rgb background) : width(w), height(h) {
        raster.resize(w * h);
        std::fill(raster.begin(), raster.end(), background);
    }

    inline void set(int x, int y, const rgb &color) {
        raster[x * width + y] = color;
    }

    inline rgb& get(int x, int y) {
        return raster[x * width + y];
    }

    inline const rgb& get(int x, int y) const {
        return raster[x * width + y];
    }

    void gammaCorrect(double gamma) {
        rgb temp;
        double power = 1.0 / gamma;
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                temp = get(i, j);
                set(i, j, rgb{pow(temp.r, power), 
                              pow(temp.g, power),
                              pow(temp.b, power)});
            }
        }
    }

    void writePPM(std::ostream &);
    void readPPM(std::istream &) const;
};