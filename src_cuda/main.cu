#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <chrono>

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

__global__ void setup(Scene* scene) {
    scene->init();
    scene->addTriangle({  0, 500, -1000}, {250, 250, -1000}, {500, 500, -1000}); // top
    scene->addTriangle({  0, 500, -1000}, {250, 250, -1000}, {0,     0, -1000}); // left
    scene->addTriangle({500,   0, -1000}, {250, 250, -1000}, {500, 500, -1000}); // right
    scene->addTriangle({500,   0, -1000}, {250, 250, -1000}, {0,     0, -1000}); // bottom
}

__global__ void render(rgb* fb, int nx, int ny, Scene* scene) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) return;
    Ray r = Ray{{float(i), float(j), 0}, {0,0,-1}};
    Hit hit;
    size_t pixel_index = j*nx + i;
    if (scene->intersect(r, hit))
        fb[pixel_index] = hit.color;
    else fb[pixel_index] = {.2,.2,.2};
}

int main() {
    int nx = 500;
    int ny = 500;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(rgb);

    rgb *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    std::chrono::time_point<std::chrono::system_clock> start, stop;
    start = std::chrono::system_clock::now();

    Scene* scene;
    cudaMalloc((void**)&scene, sizeof(Scene));

    setup<<<1, 1>>>(scene);

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    render<<<blocks, threads>>>(fb, nx, ny, scene);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    stop = std::chrono::system_clock::now();
    std::cerr << "took " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()  << " ns, "
                         << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms, "
                         << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count()      << " s\n";

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].r);
            int ig = int(255.99*fb[pixel_index].g);
            int ib = int(255.99*fb[pixel_index].b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(fb));
    cudaDeviceReset();

    return 0;
}