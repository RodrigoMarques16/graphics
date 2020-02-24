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

void random(float samples, int num_samples);
void jitter(float samples, int num_samples);
void shuffle(float samples, int num_samples);

constexpr float solve(float r) {
    auto u = r;
    for(int i = 0; i < 5;++i)
        u = (11.0f * r + u * u * (6.0f + u * (8.0f - 9.0f * u))) /
            (4.0f + 12.0f * u * (1.0f + u * (1.0f - u)));
    return u;
}

constexpr float cubicFilter(float x) {
    if (x < 1.0f / 24.0f)
        return powf(24 * x, 0.25f) - 2.0f;
    if (x < 0.5f)
        return solve(24.0f * (x - 1.0f / 24.0f) / 11.0f) - 1.0f;
    if (x < 23.0f / 24.0f)
        return 1.0f - solve(24.0f * (23.0f / 24.0f - x) / 11.0f);
    return 2 - powf(24.0f * (1.0f - x), 0.25f);
}

} // namespace sampling