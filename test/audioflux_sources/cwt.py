"""audioFlux CWT, synsq and WSST reference outputs (test/audioflux_files/cwt/).

Julia counterpart: test/cwt_af.jl. All transforms run on the first 2^12
samples of test.wav (audioFlux needs a power-of-two length).
"""
import numpy as np
import audioflux as af
from audioflux.type import SpectralFilterBankScaleType as Scale, WaveletContinueType as W
from _common import load, save

x, sr = load(2 ** 12)

# cwt_<wavelet>: af.CWT(num=84, radix2_exp=12, samplate=sr, low_fre=32.703, bin_per_octave=12,
#                       wavelet_type=W.<wavelet>, scale_type=Scale.OCTAVE, is_padding=<pad>); obj.cwt(x)
for name in ["MORSE", "MORLET", "BUMP", "PAUL", "DOG", "MEXICAN", "HERMIT", "RICKER"]:
    for pad in (False, True):
        obj = af.CWT(num=84, radix2_exp=12, samplate=sr, low_fre=32.703, bin_per_octave=12,
                     wavelet_type=getattr(W, name), scale_type=Scale.OCTAVE, is_padding=pad)
        spec = obj.cwt(x)
        save("cwt", f"cwt_{name.lower()}_{'pad' if pad else 'nopad'}", spec=spec,
             freq=obj.get_fre_band_arr())

# other frequency grids with the Morse wavelet, no padding
for name, num, low, high in [("MEL", 64, 0.0, 8000.0), ("BARK", 40, 0.0, 8000.0),
                             ("ERB", 40, 0.0, 8000.0), ("LINSPACE", 40, 100.0, 7000.0),
                             ("LOG", 60, 100.0, 7000.0)]:
    obj = af.CWT(num=num, radix2_exp=12, samplate=sr, low_fre=low, high_fre=high, bin_per_octave=12,
                 wavelet_type=W.MORSE, scale_type=getattr(Scale, name), is_padding=False)
    spec = obj.cwt(x)
    save("cwt", f"cwt_morse_{name.lower()}", spec=spec, freq=obj.get_fre_band_arr(),
         num=num, low=low, high=high)

# synsq on a Morse CWT (octave and mel grids):
# s = af.Synsq(num=obj.num, radix2_exp=12, samplate=sr)
# out = s.synsq(cwt, filter_bank_type=obj.scale_type, fre_arr=obj.get_fre_band_arr())
for name, num, low in [("OCTAVE", 84, 32.703), ("MEL", 64, 0.0)]:
    obj = af.CWT(num=num, radix2_exp=12, samplate=sr, low_fre=low, high_fre=8000.0, bin_per_octave=12,
                 wavelet_type=W.MORSE, scale_type=getattr(Scale, name), is_padding=False)
    spec = obj.cwt(x)
    s = af.Synsq(num=num, radix2_exp=12, samplate=sr)
    out = s.synsq(spec, filter_bank_type=getattr(Scale, name), fre_arr=obj.get_fre_band_arr())
    save("cwt", f"synsq_{name.lower()}", out=out, cwt=spec, freq=obj.get_fre_band_arr())

# wsst: af.WSST(num=num, radix2_exp=12, samplate=sr, low_fre=low, high_fre=8000, bin_per_octave=12,
#               wavelet_type=W.MORSE, scale_type=..., thresh=0.001, is_padding=False)
# out, cwt = obj.wsst(x)
for name, num, low in [("OCTAVE", 84, 32.703), ("MEL", 64, 0.0)]:
    obj = af.WSST(num=num, radix2_exp=12, samplate=sr, low_fre=low, high_fre=8000.0, bin_per_octave=12,
                  wavelet_type=W.MORSE, scale_type=getattr(Scale, name), thresh=0.001, is_padding=False)
    out, spec = obj.wsst(x)
    save("cwt", f"wsst_{name.lower()}", out=out, cwt=spec, freq=obj.get_fre_band_arr())
