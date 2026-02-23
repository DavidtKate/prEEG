#pragma once
#include <string>
#include <vector>
#include <memory>

struct OrtxClassifier {
    OrtxClassifier(const std::wstring& model_path);
    ~OrtxClassifier();

    std::vector<float> predict_proba(const std::vector<float>& features);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};