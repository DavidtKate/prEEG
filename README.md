# prEEG: 5-Class EEG Band Classifier (Synthetic Demo)

This project demonstrates a small end-to-end EEG-like inference pipeline:

- Read a 1-column CSV signal (`N=1024`, 4 s at 256 Hz)
- Compute relative bandpower features:
  - `delta`, `theta`, `alpha`, `beta`, `gamma` (approximately summing to 1)
- Run ONNX Runtime inference from a scikit-learn MLP exported to ONNX
- Print 5-class probabilities and benchmark latency

Classes:
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
- `model/eeg_mlp_5class.onnx`

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
.\build\Release\eeg_infer.exe --model model/eeg_mlp_5class.onnx --csv data/tests/alpha_dominant.csv --fs 256 --benchmark 200
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
- Replace synthetic generation in `scripts/train_export_onnx.py` with real EEG data + labels for real-world use.
- Keep feature extraction consistent between training and inference (same band definitions and preprocessing).
