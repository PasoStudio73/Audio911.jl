"""audioFlux pitch estimators (test/audioflux_files/pitch/).

Julia counterpart: test/pitch_af.jl. Frames of 1024 samples, hop 256,
range 50-500 Hz, on test.wav and on a synthetic harmonic tone sweep.
"""
import numpy as np
import audioflux as af
from audioflux.type import WindowType
from _common import load, save

x, sr = load()
t = np.arange(2 * sr) / sr
f0 = 110 * 2 ** (t / 2)                                     # 110 → 220 Hz over 2 s
ph = 2 * np.pi * np.cumsum(f0) / sr
tone = np.ascontiguousarray(sum(np.sin(k * ph) / k for k in range(1, 6)) * 0.3, dtype=np.float32)

kw = dict(samplate=sr, low_fre=50.0, high_fre=500.0, radix2_exp=10, slide_length=256)
out = {"tone": tone}
for name, sig in (("wav", x), ("tone", tone)):
    # af.PitchPEF(samplate=sr, low_fre=50, high_fre=500, cut_fre=4000, radix2_exp=10,
    #             slide_length=256, window_type=HAMM, alpha=10, beta=0.5, gamma=1.8)
    out[f"pef_{name}"] = af.PitchPEF(cut_fre=4000.0, window_type=WindowType.HAMM, alpha=10, beta=0.5, gamma=1.8, **kw).pitch(sig)
    # af.PitchHPS(..., window_type=HAMM, harmonic_count=5), af.PitchLHS(...)
    out[f"hps_{name}"] = af.PitchHPS(window_type=WindowType.HAMM, harmonic_count=5, **kw).pitch(sig)
    out[f"lhs_{name}"] = af.PitchLHS(window_type=WindowType.HAMM, harmonic_count=5, **kw).pitch(sig)
    # af.PitchNCF(..., window_type=RECT), af.PitchCEP(..., window_type=HAMM), af.PitchSTFT(..., HAMM)
    out[f"ncf_{name}"] = af.PitchNCF(window_type=WindowType.RECT, **kw).pitch(sig)
    out[f"cep_{name}"] = af.PitchCEP(window_type=WindowType.HAMM, **kw).pitch(sig)
    out[f"stft_{name}"] = af.PitchSTFT(window_type=WindowType.HAMM, **kw).pitch(sig)
    # af.PitchYIN(samplate=sr, low_fre=50, high_fre=500, radix2_exp=10, slide_length=256, auto_length=512)
    f, v1, v2 = af.PitchYIN(samplate=sr, low_fre=50.0, high_fre=500.0, radix2_exp=10, slide_length=256,
                            auto_length=512).pitch(sig)
    out[f"yin_{name}"] = f
save("pitch", "pitch", **out)
