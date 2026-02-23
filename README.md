# prEEG: EEG Dominant Band Classifier (C++/ONNX)

C++ EEG dominant band classifier using ONNX Runtime for real-time inference.

- Reads a 1-column CSV signal (`N=1024`, 4 s at 256 Hz)
- Computes relative bandpower features:
  - `delta`, `theta`, `alpha`, `beta`, `gamma` (~equals 1)
- Runs ONNX Runtime inference from a scikit-learn MLP exported to ONNX
- Prints 5-class probabilities and benchmarks latency

## Dependencies
### C++ Runtime
- C++17 compatible compiler (tested with MSVC / Visual Studio 2022)
- CMake ≥ 3.20
- ONNX Runtime (CPU build), https://onnxruntime.ai/
You must provide ONNXRUNTIME_ROOT pointing to the extracted ONNX Runtime directory.
### Python
- Python ≥ 3.9
- numpy
- scikit-learn
- skl2onnx
Install Python dependencies:
```powershell
pip install numpy scikit-learn skl2onnx
```

## Classes
- `0=delta`
- `1=theta`
- `2=alpha`
- `3=beta`
- `4=gamma`

## Build (Windows, VS2022, CMake)

Set your ONNX Runtime root and build:

```powershell
cmake -S . -B build -DONNXRUNTIME_ROOT=C:/libs/onnxruntime-win-x64-1.24.2
cmake --build build --config Release
```

## Train + Export ONNX (5 classes)

```powershell
python scripts/train_export_onnx.py
```

This writes:
- `model/eeg_mlp.onnx`

Export uses:
- `options={id(pipe): {"zipmap": False}}`

so class probabilities are emitted as a plain float tensor (compatible with C++ ONNX Runtime tensor parsing).

## Generate Test CSVs

```powershell
python scripts/make_test_csvs.py
```

Generated under `data/tests/`:
- `delta_dominant.csv` (~2 Hz)
- `theta_dominant.csv` (~6 Hz)
- `alpha_dominant.csv` (~10 Hz)
- `beta_dominant.csv` (~20 Hz)
- `gamma_dominant.csv` (~40 Hz)
- `mixed_alpha_beta.csv`
- `noisy.csv`

## Run Inference

```powershell
.\build\Release\eeg_infer.exe --model model/eeg_mlp.onnx --csv data/tests/alpha_dominant.csv --fs 256 --benchmark 200
```

Example output shape:

```text
Features (rel power): delta=... theta=... alpha=... beta=... gamma=...
Probs: delta=... theta=... alpha=... beta=... gamma=... -> pred=alpha (2)
Benchmark: avg=... ms, p95=... ms (200 runs)
```

If uncertain or mixed input:

```text
Warning: Low confidence / mixed spectrum (top_prob=..., dominance_ratio=...)
```

## Notes

- Training data in this repo is synthetic and intended for demonstration only.
- Replace the synthetic generation in `scripts/train_export_onnx.py` with real EEG data + labels for real-world use.
