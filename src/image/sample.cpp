#include "sample.hpp"
#include "vec2.hpp"
#include <stdlib.h>

namespace sampling {

void random(Vec2 *samples, int num_samples) {
    for (int i = 0; i < num_samples; ++i) {
        samples[i].x = drand48();
        samples[i].y = drand48();
    }
}

void jitter(Vec2* samples, int num_samples) {
    
}

} // namespace sampling
