#pragma once

#include "math/rgb.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

struct Image {
    int width, height;
    rgb* raster;

    Image() = default;
    
    Image(int w, int h) : width(w), height(h) {
        cudaMallocManaged((void**)&raster, w*h*sizeof(rgb));
    }
    
    Image(int w, int h, rgb background) : width(w), height(h) {
        cudaMallocManaged((void**)&raster, w*h*sizeof(rgb));
        for(int i = 0; i < w * h; ++i) 
            raster[i] = background;
    }

    ~Image() {
        cudaFree(raster);
    }
    
    __host__ __device__ inline void set(int x, int y, const rgb &color) {
        raster[x * width + y] = color;
    }

    __host__ __device__ inline rgb& get(int x, int y) {
        return raster[x * width + y];
    }

    __host__ __device__ inline const rgb& get(int x, int y) const {
        return raster[x * width + y];
    }

    void gammaCorrect(double gamma) {
        rgb temp;
        double power = 1.0 / gamma;
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                temp = get(i, j);
                set(i, j, rgb{powf(temp.r, power), 
                              powf(temp.g, power),
                              powf(temp.b, power)});
            }
        }
    }

    void writePPM(std::ostream& os) {
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
    
    void readPPM(std::istream &) const;
};