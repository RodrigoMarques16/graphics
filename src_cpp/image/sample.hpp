#pragma once

#include <cmath>

struct Vec2;

namespace sampling {

void random(Vec2* samples, int num_samples);
void jitter(Vec2* samples, int num_samples);
void nrooks(Vec2* samples, int num_samples);
void multiJitter(Vec2* samples, int num_samples);
void shuffle(Vec2* samples, int num_samples);

void boxFilter(Vec2* samples, int num_samples);
void tentFilter(Vec2* samples, int num_samples);
void cubicSplineFilter(Vec2* samples, int num_samples);

void random(double samples, int num_samples);
void jitter(double samples, int num_samples);
void shuffle(double samples, int num_samples);

constexpr double solve(double r) {
    auto u = r;
    for(int i = 0; i < 5;++i)
        u = (11.0 * r + u * u * (6.0 + u * (8.0 - 9.0 * u))) /
            (4.0 + 12.0 * u * (1.0 + u * (1.0 - u)));
    return u;
}

constexpr double cubicFilter(double x) {
    if (x < 1.0 / 24.0)
        return powf(24 * x, 0.25) - 2.0;
    if (x < 0.5)
        return solve(24.0 * (x - 1.0 / 24.0) / 11.0) - 1.0;
    if (x < 23.0 / 24.0)
        return 1.0 - solve(24.0 * (23.0 / 24.0 - x) / 11.0);
    return 2 - powf(24.0 * (1.0 - x), 0.25);
}

} // namespace sampling