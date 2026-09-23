"""audioFlux non-negative matrix factorisation (test/audioflux_files/nmf/).

Julia counterpart: test/nmf_af.jl. The magnitude STFT of the first second
of test.wav (Hann 512, hop 256) is factorised into 4 components with every
divergence (tp 0 KL, 1 IS, 2 Euclidean) and column norm (0 max, 1 L1,
2 L2), from audioFlux's ramp initialisation. X is saved so both sides
factorise the same float32 matrix.
"""
import numpy as np
import audioflux as af
from audioflux.classic.nmf import nmf as af_nmf
from audioflux.type import WindowType
from _common import load, save

x, sr = load()
x = x[:sr]
S = af.STFT(radix2_exp=9, window_type=WindowType.HANN, slide_length=256).stft(x)
X = np.ascontiguousarray(np.abs(S), dtype=np.float32)
out = {"X": X}
# h, w = audioflux.classic.nmf.nmf(X, 4, max_iter=<iters>, tp=<tp>, thresh=1e-3, norm=<norm>)
for tp in (0, 1, 2):
    for norm in (0, 1, 2):
        for iters in (20, 300):
            h, w = af_nmf(X, 4, max_iter=iters, tp=tp, thresh=1e-3, norm=norm)
            out[f"w_{tp}_{norm}_{iters}"] = w
            out[f"h_{tp}_{norm}_{iters}"] = h
save("nmf", "nmf", **out)
