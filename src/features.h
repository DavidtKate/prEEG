#pragma once
#include <vector>

struct BandFeatures {
    float delta, theta, alpha, beta, gamma;
};

BandFeatures bandpower_relative(const std::vector<float>& x, float fs);
std::vector<float> read_csv_1col(const char* path);