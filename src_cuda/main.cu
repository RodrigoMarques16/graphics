#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "math/ray.cuh"
#include "image.cuh"
#include "render/scene.cuh"
#include "math/rgb.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void render(rgb* fb, int nx, int ny, const Scene& scene) {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            Ray r = Ray{{i, j, 0}, {0,0,-1}};
            Hit* hit = scene.intersect(r);   
            if (hit) {
                fb[j*nx+i] = hit->color;
            } else {
                fb[j*nx+i] = {.2,.2,.2};
            }
        }
    }
}

int main() {
    int nx = 500;
    int ny = 500;
    int tx = 10;
    int ty = 10;

    // Image img(nx, ny, {.2, .2, .2});
    Scene scene = Scene();

    rgb* fb;
    cudaMallocManaged((void**)&fb, nx * ny);
    
    //                     BLUE                RED               GREEN
    scene.addTriangle({  0, 500, -1000}, {250, 250, -1000}, {500, 500, -1000}); // top
    scene.addTriangle({  0, 500, -1000}, {250, 250, -1000}, {0,     0, -1000}); // left
    scene.addTriangle({500,   0, -1000}, {250, 250, -1000}, {500, 500, -1000}); // right
    scene.addTriangle({500,   0, -1000}, {250, 250, -1000}, {0,     0, -1000}); // bottom
    printf("hi\n");
    
    // Sphere s = {{250, 250, -1000}, 150*150};
    // Triangle t = {{300, 600, -800}, {0, 100, -1000}, {450, 20, -1000}};
    // scene.addTriangle(t);
    // scene.addSphere(s);


    dim3 blocks(nx/tx+1, ny/ty+1);
    dim3 threads(tx,ty);
    fprintf(stderr, "nx: %d, ny: %d, blocks: %d, threads: %d\n", nx, ny, blocks, threads);
    render<<<1, 1>>>(fb, nx, ny, scene);

    cudaDeviceSynchronize();

    std::cout << "P6\n" << nx << ' ' << ny << '\n' << "255\n";
        for (int i = nx - 1; i >= 0; --i) {
            for (int j = 0; j < ny; ++j) {
                rgb color   = fb[i*nx+j];
                uint ired   = std::min(255u, (unsigned int) (256u * color.r));
                uint igreen = std::min(255u, (unsigned int) (256u * color.g));
                uint iblue  = std::min(255u, (unsigned int) (256u * color.b));
                std::cout << (unsigned char) ired
                   << (unsigned char) igreen
                   << (unsigned char) iblue;
            }
        }
}