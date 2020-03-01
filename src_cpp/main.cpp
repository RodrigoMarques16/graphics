#include "onb.hpp"
#include "image.hpp"
#include "rgb.hpp"
#include "unit3.hpp"
#include "vec3.hpp"
#include "ray.hpp"
#include "triangle.hpp"
#include "sphere.hpp"
#include "scene.hpp"
#include <iostream>
#include <stdio.h>


int main() {

    Image img(500, 500, {.2, .2, .2});

    Scene scene;
    //                     BLUE                RED               GREEN
    scene.addTriangle({  0, 500, -1000}, {250, 250, -1000}, {500, 500, -1000}); // top
    scene.addTriangle({  0, 500, -1000}, {250, 250, -1000}, {0,     0, -1000}); // left
    scene.addTriangle({500,   0, -1000}, {250, 250, -1000}, {500, 500, -1000}); // right
    scene.addTriangle({500,   0, -1000}, {250, 250, -1000}, {0,     0, -1000}); // bottom
    
    // Sphere s = {{250, 250, -1000}, 150*150};
    // Triangle t = {{300, 600, -800}, {0, 100, -1000}, {450, 20, -1000}};
    // scene.addTriangle(t);
    // scene.addSphere(s);

    for (double i = 0; i < 500; i++) {
        for (double j = 0; j < 500; j++) {
            auto r = Ray{{i, j, 0}, Vec3::back}; // todo: generate ray based on camera
            auto hit = scene.intersect(r);       // customization point   
            if (hit.has_value())                 
                img.set(i,j, hit->color);        // todo: add hit shader
                                                 // todo: add miss shader
        }
    }


    
    img.writePPM(std::cout);
}