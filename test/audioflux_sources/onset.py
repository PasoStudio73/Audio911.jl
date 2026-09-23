"""audioFlux onset detection (test/audioflux_files/onset/).

Julia counterpart: test/onset_af.jl. Every novelty type on the magnitude
STFT of test.wav (periodic Hann 512, hop 256) and its phase.
"""
import numpy as np
import audioflux as af
from audioflux.type import WindowType, NoveltyType
from audioflux.mir.onset import NoveltyParam
from ctypes import c_int, c_float
from _common import load, save

x, sr = load()
stft = af.STFT(radix2_exp=9, window_type=WindowType.HANN, slide_length=256)
S = stft.stft(x)
mag = np.ascontiguousarray(np.abs(S), dtype=np.float32)
phase = np.ascontiguousarray(np.angle(S), dtype=np.float32)
n_fre, n_time = mag.shape
out = {}
# obj = af.Onset(time_length=T, fre_length=257, slide_length=256, samplate=sr,
#                filter_order=<order>, novelty_type=NoveltyType.<type>)
# point, evn, time, value = obj.onset(mag, phase)
for name, order in [("FLUX", 1), ("HFC", 1), ("SD", 1), ("SF", 1), ("MKL", 1), ("PD", 1), ("WPD", 1),
                    ("NWPD", 1), ("CD", 1), ("RCD", 1), ("BROADBAND", 1), ("FLUX", 3), ("SD", 5)]:
    obj = af.Onset(time_length=n_time, fre_length=n_fre, slide_length=256, samplate=sr,
                   filter_order=order, novelty_type=getattr(NoveltyType, name))
    point, evn, t, v = obj.onset(mag, phase)
    key = f"{name.lower()}_{order}"
    out["evn_" + key] = evn
    out["point_" + key] = np.asarray(point, dtype=np.float64) + 1     # 1-based
save("onset", "onset", **out)
