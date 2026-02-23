import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skl2onnx import to_onnx

np.random.seed(1)

FS = 256    # Hz
N = FS * 4  # seconds

BANDS = {
    0: ("delta", (1.0, 4.0)),
    1: ("theta", (4.0, 8.0)),
    2: ("alpha", (8.0, 13.0)),
    3: ("beta",  (13.0, 30.0)),
    4: ("gamma", (30.0, 45.0)),
}

def bandpower_features(x: np.ndarray, fs: int):
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1/fs)
    psd = (np.abs(X) ** 2) / x.size

    def bp(f_lo, f_hi):
        m = (freqs >= f_lo) & (freqs < f_hi)
        return float(psd[m].sum())

    delta = bp(1, 4)
    theta = bp(4, 8)
    alpha = bp(8, 13)
    beta  = bp(13, 30)
    gamma = bp(30, 45)
    total = bp(1, 45) + 1e-9

    feats = np.array([delta, theta, alpha, beta, gamma], dtype=np.float32) / total
    return feats

def synth_epoch(dominant_class: int):
    t = np.arange(N) / FS

    name, (lo, hi) = BANDS[dominant_class]
    dom_freq = np.random.uniform(lo, hi)

    # Dominant component
    dom_amp = np.random.uniform(1.0, 1.6)
    x = dom_amp * np.sin(2*np.pi*dom_freq*t)

    # Add some weaker components from random bands
    k = np.random.randint(1, 4)
    for _ in range(k):
        c = np.random.randint(0, 5)
        f_lo, f_hi = BANDS[c][1]
        f = np.random.uniform(f_lo, f_hi)
        amp = np.random.uniform(0.05, 0.35)
        x += amp * np.sin(2*np.pi*f*t + np.random.uniform(0, 2*np.pi))

    # Add noise
    noise = np.random.uniform(0.3, 1.0)
    x += noise * np.random.randn(N)

    return x.astype(np.float32)

# Build dataset
M = 5000
X = np.zeros((M, 5), dtype=np.float32)
y = np.zeros((M,), dtype=np.int64)

for i in range(M):
    label = i % 5
    sig = synth_epoch(label)
    X[i] = bandpower_features(sig, FS)
    y[i] = label

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=0))
])

pipe.fit(Xtr, ytr)
acc = pipe.score(Xte, yte)
print("Test accuracy:", acc)

# Export with zipmap disabled so probabilities are a tensor
options = {id(pipe): {"zipmap": False}}
onnx_model = to_onnx(pipe, Xtr[:1].astype(np.float32), target_opset=17, options=options)

Path("model").mkdir(parents=True, exist_ok=True)
with open("model/eeg_mlp.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Wrote model/eeg_mlp.onnx")
