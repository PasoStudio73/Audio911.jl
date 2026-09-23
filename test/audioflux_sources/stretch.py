"""audioFlux time stretch, pitch shift and resampling (test/audioflux_files/stretch/).

Julia counterpart: test/stretch_af.jl. TimeStretch and PitchShift keep
their defaults (Hann, fft 4096, hop 1024); Resample and WindowResample are
run for the three qualities and a Hann window, down and up.
"""
import numpy as np
import audioflux as af
from audioflux.type import WindowType, ResampleQualityType
from _common import load, save

x, sr = load()
out = {"sr": sr}
# obj = af.TimeStretch(radix2_exp=12, slide_length=1024, window_type=WindowType.HANN)
# y = obj.time_stretch(x, rate)   (capacity ceil(n / rate) + 4096; the C call returns round(n / rate))
ts = af.TimeStretch(radix2_exp=12, slide_length=1024, window_type=WindowType.HANN)
for name, rate in (("0_5", 0.5), ("0_8", 0.8), ("1_5", 1.5)):
    out["ts_" + name] = ts.time_stretch(x, rate)
# obj = af.PitchShift(radix2_exp=12, slide_length=1024, window_type=WindowType.HANN)
# y = obj.pitch_shift(x, n_semitone, samplate=sr)
ps = af.PitchShift(radix2_exp=12, slide_length=1024, window_type=WindowType.HANN)
for n in (-5, 3, 7):
    out[f"ps_{n}".replace("-", "m")] = ps.pitch_shift(x, n, samplate=sr)
# obj = af.Resample(qual_type, is_scale); obj.set_samplate(sr, target); y = obj.resample(x)
for q in ("BEST", "MID", "FAST"):
    for target, scale in ((sr // 2, False), (int(sr * 1.5), True), (11025, False)):
        obj = af.Resample(qual_type=getattr(ResampleQualityType, q), is_scale=scale)
        obj.set_samplate(sr, target)
        out[f"rs_{q.lower()}_{target}"] = obj.resample(x)
# obj = af.WindowResample(zero_num=32, nbit=9, win_type=WindowType.HANN, value=0, roll_off=0.9)
obj = af.WindowResample(zero_num=32, nbit=9, win_type=WindowType.HANN, value=0, roll_off=0.9)
obj.set_samplate(sr, 12000)
out["wrs_hann_12000"] = obj.resample(x)
save("stretch", "stretch", **out)
