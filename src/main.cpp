#include "features.h"
#include "onnx_infer.h"
#include <array>
#include <iostream>
#include <chrono>
#include <algorithm>

static std::wstring to_w(const std::string& s) {
    return std::wstring(s.begin(), s.end());
}

int main(int argc, char** argv) {
    std::string model_path, csv_path;
    float fs = 256.f;
    int bench = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i+1 < argc) model_path = argv[++i];
        else if (a == "--csv" && i+1 < argc) csv_path = argv[++i];
        else if (a == "--fs" && i+1 < argc) fs = std::stof(argv[++i]);
        else if (a == "--benchmark" && i+1 < argc) bench = std::stoi(argv[++i]);
        else {
            std::cerr << "Usage: eeg_infer --model model.onnx --csv sample.csv --fs 256 [--benchmark 200]\n";
            return 1;
        }
    }

    if (model_path.empty() || csv_path.empty()) {
        std::cerr << "Missing --model or --csv\n";
        return 1;
    }

    auto x = read_csv_1col(csv_path.c_str());
    auto bf = bandpower_relative(x, fs);
    std::vector<float> feats = {bf.delta, bf.theta, bf.alpha, bf.beta, bf.gamma};

    std::cout << "Features (rel power): "
              << "delta=" << bf.delta << " theta=" << bf.theta
              << " alpha=" << bf.alpha << " beta=" << bf.beta
              << " gamma=" << bf.gamma << "\n";

    OrtxClassifier clf(to_w(model_path));

    auto probs = clf.predict_proba(feats);
    if (probs.size() < 5) {
        std::cerr << "Model probability output must contain 5 classes, got " << probs.size() << "\n";
        return 2;
    }

    static const std::array<const char*, 5> kLabels = {"delta", "theta", "alpha", "beta", "gamma"};
    auto pred_it = std::max_element(probs.begin(), probs.begin() + 5);
    const int pred = static_cast<int>(pred_it - probs.begin());
    const float top_prob = *pred_it;

    std::cout << "Probs: ";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << kLabels[i] << "=" << probs[i];
        if (i + 1 < 5) std::cout << " ";
    }
    std::cout << " -> pred=" << kLabels[pred] << " (" << pred << ")\n";

    std::array<float, 5> bp = {bf.delta, bf.theta, bf.alpha, bf.beta, bf.gamma};
    std::sort(bp.begin(), bp.end(), std::greater<float>());
    const float dominance_ratio = bp[0] / (bp[1] + 1e-9f);
    if (top_prob < 0.6f || dominance_ratio < 1.5f) {
        std::cout << "Warning: Low confidence / mixed spectrum"
                  << " (top_prob=" << top_prob
                  << ", dominance_ratio=" << dominance_ratio << ")\n";
    }

    if (bench > 0) {
        for (int i = 0; i < 10; ++i) clf.predict_proba(feats);

        std::vector<double> ms;
        ms.reserve(bench);
        for (int i = 0; i < bench; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            (void)clf.predict_proba(feats);
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
            ms.push_back(dt);
        }
        double sum = 0;
        for (double v : ms) sum += v;
        double avg = sum / ms.size();
        std::sort(ms.begin(), ms.end());
        double p95 = ms[(size_t)(0.95 * (ms.size()-1))];

        std::cout << "Benchmark: avg=" << avg << " ms, p95=" << p95 << " ms (" << bench << " runs)\n";
    }

    return 0;
}
