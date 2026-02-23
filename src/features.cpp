#include "features.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>

static constexpr float PI = 3.14159265358979323846f;

std::vector<float> read_csv_1col(const char* path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Failed to open CSV");
    std::vector<float> x;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        if (std::getline(ss, cell, ',')) {
            x.push_back(std::stof(cell));
        }
    }
    return x;
}

static void dft_rfft_power(const std::vector<float>& x, std::vector<float>& freqs, std::vector<float>& power, float fs) {
    const int N = (int)x.size();
    const int K = N / 2 + 1;
    freqs.resize(K);
    power.resize(K);

    for (int k = 0; k < K; ++k) {
        float re = 0.f, im = 0.f;
        for (int n = 0; n < N; ++n) {
            float ang = 2.f * PI * (float)k * (float)n / (float)N;
            re += x[n] * std::cos(ang);
            im -= x[n] * std::sin(ang);
        }
        float mag2 = re*re + im*im;
        power[k] = mag2 / (float)N;
        freqs[k] = (fs * (float)k) / (float)N;
    }
}

static float band_sum(const std::vector<float>& freqs, const std::vector<float>& p, float lo, float hi) {
    float s = 0.f;
    for (size_t i = 0; i < freqs.size(); ++i) {
        if (freqs[i] >= lo && freqs[i] < hi) s += p[i];
    }
    return s;
}

BandFeatures bandpower_relative(const std::vector<float>& x, float fs) {
    std::vector<float> freqs, p;
    dft_rfft_power(x, freqs, p, fs);

    float delta = band_sum(freqs, p, 1.f, 4.f);
    float theta = band_sum(freqs, p, 4.f, 8.f);
    float alpha = band_sum(freqs, p, 8.f, 13.f);
    float beta  = band_sum(freqs, p, 13.f, 30.f);
    float gamma = band_sum(freqs, p, 30.f, 45.f);
    float total = band_sum(freqs, p, 1.f, 45.f) + 1e-9f;

    BandFeatures bf;
    bf.delta = delta / total;
    bf.theta = theta / total;
    bf.alpha = alpha / total;
    bf.beta  = beta  / total;
    bf.gamma = gamma / total;
    return bf;
}