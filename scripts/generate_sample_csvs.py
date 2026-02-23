import numpy as np
from pathlib import Path

FS = 256    # Hz
N = FS * 4  # seconds
rng = np.random.default_rng(0)


def make_signal(components, noise=0.6):
    t = np.arange(N, dtype=np.float32) / FS
    x = np.zeros_like(t, dtype=np.float32)
    for freq, amp in components:
        phase = rng.uniform(0.0, 2.0 * np.pi)
        x += (amp * np.sin(2.0 * np.pi * freq * t + phase)).astype(np.float32)
    x += (noise * rng.standard_normal(N)).astype(np.float32)
    return x


out = Path("data/tests")
out.mkdir(parents=True, exist_ok=True)

signals = {
    "delta_dominant.csv": [(2.0, 1.3), (6.0, 0.2), (10.0, 0.1)],
    "theta_dominant.csv": [(6.0, 1.3), (2.0, 0.2), (10.0, 0.1)],
    "alpha_dominant.csv": [(10.0, 1.3), (6.0, 0.2), (20.0, 0.1)],
    "beta_dominant.csv": [(20.0, 1.3), (10.0, 0.2), (40.0, 0.1)],
    "gamma_dominant.csv": [(40.0, 1.1), (20.0, 0.2), (10.0, 0.1)],
    "mixed_alpha_beta.csv": [(10.0, 0.9), (20.0, 0.9), (6.0, 0.15)],
    "noisy.csv": [(10.0, 0.25), (20.0, 0.25), (40.0, 0.2)],
}

for name, comps in signals.items():
    noise = 1.25 if name == "noisy.csv" else 0.55
    x = make_signal(comps, noise=noise)
    np.savetxt(out / name, x, delimiter=",")
    print(f"Wrote {out / name}")
