#include "onnx_infer.h"
#include <onnxruntime_cxx_api.h>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>

struct OrtxClassifier::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "eeg_demo"};
    Ort::SessionOptions opts;
    Ort::Session session{nullptr};
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<Ort::AllocatedStringPtr> input_name_holders;
    std::vector<Ort::AllocatedStringPtr> output_name_holders;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    Impl(const std::wstring& model_path) {
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session = Ort::Session(env, model_path.c_str(), opts);

        const size_t input_count = session.GetInputCount();
        const size_t output_count = session.GetOutputCount();
        if (input_count == 0 || output_count == 0) {
            throw std::runtime_error("ONNX model has no inputs or outputs");
        }

        input_name_holders.push_back(session.GetInputNameAllocated(0, allocator));
        output_name_holders.push_back(session.GetOutputNameAllocated(0, allocator));
        input_names.push_back(input_name_holders.back().get());

        output_names.reserve(output_count);
        output_name_holders.reserve(output_count);
        for (size_t i = 0; i < output_count; ++i) {
            if (i > 0) output_name_holders.push_back(session.GetOutputNameAllocated(i, allocator));
            output_names.push_back(output_name_holders.back().get());
        }
    }
};

OrtxClassifier::OrtxClassifier(const std::wstring& model_path)
: impl(std::make_unique<Impl>(model_path)) {}

OrtxClassifier::~OrtxClassifier() = default;

std::vector<float> OrtxClassifier::predict_proba(const std::vector<float>& features) {
    if (features.size() != 5) throw std::runtime_error("Expected 5 features");

    std::array<int64_t, 2> shape{1, 5};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // Input tensor
    std::vector<float> input = features;
    Ort::Value x = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), shape.data(), shape.size());

    auto outputs = impl->session.Run(
        Ort::RunOptions{nullptr},
        impl->input_names.data(), &x, 1,
        impl->output_names.data(), impl->output_names.size()
    );

    // Find probability output tensor.
    size_t best_idx = outputs.size();
    int best_score = -1;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const Ort::Value& out = outputs[i];
        if (!out.IsTensor()) continue;

        Ort::TypeInfo ti = out.GetTypeInfo();
        auto tinfo = ti.GetTensorTypeAndShapeInfo();
        if (tinfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) continue;

        const size_t count = tinfo.GetElementCount();
        if (count < 2) continue;

        std::string name = impl->output_names[i] ? impl->output_names[i] : "";
        for (char& ch : name) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        const bool looks_prob = (name.find("prob") != std::string::npos);
        const bool five_class = (count == 5);

        int score = 0;
        if (looks_prob) score += 4;
        if (five_class) score += 3;
        if (count >= 5) score += 2;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx == outputs.size()) {
        throw std::runtime_error("Could not find probability float tensor output");
    }

    const Ort::Value& best = outputs[best_idx];
    auto tinfo = best.GetTensorTypeAndShapeInfo();
    const size_t count = tinfo.GetElementCount();
    const float* data = best.GetTensorData<float>();

    const size_t want = (count >= 5) ? 5 : count;
    return std::vector<float>(data, data + want);
}
