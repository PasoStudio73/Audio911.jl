"""audioFlux PWT, ST, FST and NSGT reference outputs (test/audioflux_files/transforms/).

Julia counterpart: test/transforms.jl. The whole-signal transforms need a
power-of-two length: PWT and NSGT use the first 2^14 samples, ST and FST the
first 2^12 (the S-transform keeps an (N/2+1) x N Gaussian table).
"""
import numpy as np
import audioflux as af
from audioflux.type import (SpectralFilterBankScaleType, SpectralFilterBankStyleType,
                            SpectralFilterBankNormalType, NSGTFilterBankType)
from _common import load, save

x14, sr = load(2 ** 14)
x12, _ = load(2 ** 12)

# pwt_octave: af.PWT(num=84, radix2_exp=14, samplate=sr, low_fre=32.703, bin_per_octave=12,
#                    scale_type=OCTAVE, style_type=SLANEY, normal_type=NONE, is_padding=False)
obj = af.PWT(num=84, radix2_exp=14, samplate=sr, low_fre=32.703, high_fre=8000.0, bin_per_octave=12,
             scale_type=SpectralFilterBankScaleType.OCTAVE,
             style_type=SpectralFilterBankStyleType.SLANEY,
             normal_type=SpectralFilterBankNormalType.NONE, is_padding=False)
spec = obj.pwt(x14)
save("transforms", "pwt_octave", spec=spec, freq=obj.get_fre_band_arr())

# pwt_mel_hann: num=40, MEL scale, HANN style, BAND_WIDTH norm, 0-8000 Hz
obj = af.PWT(num=40, radix2_exp=14, samplate=sr, low_fre=0.0, high_fre=8000.0, bin_per_octave=12,
             scale_type=SpectralFilterBankScaleType.MEL,
             style_type=SpectralFilterBankStyleType.HANN,
             normal_type=SpectralFilterBankNormalType.BAND_WIDTH, is_padding=False)
spec = obj.pwt(x14)
save("transforms", "pwt_mel_hann", spec=spec, freq=obj.get_fre_band_arr())

# st: af.ST(radix2_exp=12, min_index=1, max_index=1024, samplate=sr, factor=1., norm=1.)
obj = af.ST(radix2_exp=12, min_index=1, max_index=1024, samplate=sr, factor=1.0, norm=1.0)
spec = obj.st(x12)
save("transforms", "st", spec=spec, min_index=1, max_index=1024)

# st_factor: factor=0.5, norm=1.2, min_index=1, max_index=300
obj = af.ST(radix2_exp=12, min_index=1, max_index=300, samplate=sr, factor=0.5, norm=1.2)
spec = obj.st(x12)
save("transforms", "st_factor", spec=spec, min_index=1, max_index=300)

# fst: af.FST(radix2_exp=12, min_index=1, max_index=1024, samplate=sr)
obj = af.FST(radix2_exp=12, min_index=1, max_index=1024, samplate=sr)
spec = obj.fst(x12)
save("transforms", "fst", spec=spec, min_index=1, max_index=1024)

# fst_all: min_index=1, max_index=2047
obj = af.FST(radix2_exp=12, min_index=1, max_index=2047, samplate=sr)
spec = obj.fst(x12)
save("transforms", "fst_all", spec=spec, min_index=1, max_index=2047)

# nsgt_octave: af.NSGT(num=84, radix2_exp=14, samplate=sr, low_fre=32.703, bin_per_octave=12,
#                      min_len=3, nsgt_filter_bank_type=EFFICIENT, scale_type=OCTAVE,
#                      style_type=HANN, normal_type=BAND_WIDTH)
obj = af.NSGT(num=84, radix2_exp=14, samplate=sr, low_fre=32.703, high_fre=8000.0, bin_per_octave=12,
              min_len=3, nsgt_filter_bank_type=NSGTFilterBankType.EFFICIENT,
              scale_type=SpectralFilterBankScaleType.OCTAVE,
              style_type=SpectralFilterBankStyleType.HANN,
              normal_type=SpectralFilterBankNormalType.BAND_WIDTH)
spec = obj.nsgt(x14)
save("transforms", "nsgt_octave", spec=spec, freq=obj.get_fre_band_arr(),
     lengths=obj.get_time_length_arr(), max_len=obj.get_max_time_length())

# nsgt_mel_standard: num=40, MEL, STANDARD bank, HAMM style, NONE norm
obj = af.NSGT(num=40, radix2_exp=14, samplate=sr, low_fre=0.0, high_fre=8000.0, bin_per_octave=12,
              min_len=3, nsgt_filter_bank_type=NSGTFilterBankType.STANDARD,
              scale_type=SpectralFilterBankScaleType.MEL,
              style_type=SpectralFilterBankStyleType.HAMM,
              normal_type=SpectralFilterBankNormalType.NONE)
spec = obj.nsgt(x14)
save("transforms", "nsgt_mel_standard", spec=spec, freq=obj.get_fre_band_arr(),
     lengths=obj.get_time_length_arr(), max_len=obj.get_max_time_length())
