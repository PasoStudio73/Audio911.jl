"""audioFlux harmonic count (test/audioflux_files/harmonic/).

Julia counterpart: test/harmonic_af.jl. Harmonic(...).harmonic_count on
test.wav and on a synthetic harmonic tone whose number of partials changes
over time (Hamming window, fft 4096, hop 1024).
"""
import numpy as np
import audioflux as af
from audioflux.type import WindowType
from _common import load, save

x, sr = load()
out = {"sr": sr}
# obj = af.Harmonic(radix2_exp=12, samplate=sr, slide_length=1024, window_type=WindowType.HAMM,
#                   low_fre=27, high_fre=<high>); count = obj.harmonic_count(x, <low>, <high>)
high = min(4000.0, sr / 2 - 100)
obj = af.Harmonic(radix2_exp=12, samplate=sr, slide_length=1024, window_type=WindowType.HAMM,
                  low_fre=27.0, high_fre=high)
out["wav_full"] = obj.harmonic_count(x, 27.0, high)
obj = af.Harmonic(radix2_exp=12, samplate=sr, slide_length=1024, window_type=WindowType.HAMM,
                  low_fre=27.0, high_fre=high)
out["wav_band"] = obj.harmonic_count(x, 100.0, 1000.0)
out["high"] = high

# a 220 Hz tone with 3, then 6, then 10 partials (amplitude 1/k), plus faint noise
tsr = 32000
t = np.arange(3 * tsr) / tsr
y = np.zeros_like(t)
for seg, nh in enumerate((3, 6, 10)):
    m = (t >= seg) & (t < seg + 1)
    for k in range(1, nh + 1):
        y[m] += np.sin(2 * np.pi * 220 * k * t[m]) / k
y = 0.2 * y + 1e-4 * np.random.default_rng(0).standard_normal(len(y))
y = y.astype(np.float32)
obj = af.Harmonic(radix2_exp=12, samplate=tsr, slide_length=1024, window_type=WindowType.HAMM,
                  low_fre=27.0, high_fre=4000.0)
out["tone"] = y
out["tone_count"] = obj.harmonic_count(y, 27.0, 4000.0)
save("harmonic", "harmonic", **out)
