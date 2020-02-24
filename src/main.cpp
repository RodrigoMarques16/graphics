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
    Sphere s = {{250, 250, -1000}, 150*150};
    Triangle t = {{300, 600, -800}, {0, 100, -1000}, {450, 20, -1000}};

    Image img(500, 500, {.2, .2, .2});

    Scene scene;
    scene.addSphere(s);
    scene.addTriangle(t);

    for (float i = 0; i < 500; i++) {
        for (float j = 0; j < 500; j++) {
            auto r = Ray{{i, j, 0}, Vec3::back};
            auto hit = scene.intersect(r);  
            if (hit.has_value()) 
                img.set(i,j, hit->color);
        }
    }
    
    img.writePPM(std::cout);
}