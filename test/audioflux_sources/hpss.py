"""audioFlux harmonic/percussive separation (test/audioflux_files/hpss/).

Julia counterpart: test/hpss_af.jl. audioFlux's HPSS: STFT (Hamming,
fft 2048, hop forced to fft/4 = 512 whatever slide_length says), median
filters of 21 frames and 31 bins with zero-padded edges, squared soft
masks, and the weighted overlap-add ISTFT of the masked spectra.
"""
import numpy as np
import audioflux as af
from audioflux.type import WindowType
from _common import load, save

x, sr = load()
# obj = af.HPSS(radix2_exp=11, window_type=WindowType.HAMM, slide_length=512,
#               h_order=21, p_order=31); h, p = obj.hpss(x)
obj = af.HPSS(radix2_exp=11, window_type=WindowType.HAMM, slide_length=512, h_order=21, p_order=31)
h, p = obj.hpss(x)
# a second order pair (the hop stays fft/4 = 256 for radix2_exp 10)
obj2 = af.HPSS(radix2_exp=10, window_type=WindowType.HANN, slide_length=100, h_order=11, p_order=5)
h2, p2 = obj2.hpss(x)
save("hpss", "hpss", h=h, p=p, h2=h2, p2=p2, n=len(x))
