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
    Sphere s = {{250, 250, -1000}, 150, {.2,.2,.8}};
    Triangle t = {{300, 600, -800}, {0, 100, -1000}, {450, 20, -1000}, {.8,.2,.2}};

    Image img(500,500, {.2,.2,.2});

    for (float i = 0; i < 500; i++) {
        for (float j = 0; j < 500; j++) {
            auto r = Ray{{i, j, 0}, Vec3::back};
            auto tmax = 100000.0f;
            auto is_hit = false;
            auto hitSphere = s.hit(r, 0.00001f, tmax);
            HitRecord nearest;
            if (hitSphere.has_value()) {
                tmax = hitSphere.value().dist;
                is_hit = true;
                nearest = hitSphere.value();
            }
            auto hitTriangle = t.hit(r, 0.00001f, tmax);
            if(hitTriangle.has_value()) {
                is_hit = true;
                nearest = hitTriangle.value();
            }
            if (is_hit)
                img.set(i, j, nearest.color);
        }
    }
    img.writePPM(std::cout);
}