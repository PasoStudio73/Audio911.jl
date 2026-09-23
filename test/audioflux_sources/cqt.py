"""audioFlux CQT / VQT reference outputs (test/audioflux_files/cqt/).

Julia counterpart: test/cqt.jl. Every call is repeated as a comment there.
"""
import numpy as np
import audioflux as af
from audioflux.type import WindowType, SpectralFilterBankNormalType
from _common import load, save

x, sr = load()

# cqt_area: obj = af.CQT(num=84, samplate=sr, low_fre=32.703, bin_per_octave=12,
#                        factor=1.0, beta=0.0, thresh=0.01, window_type=WindowType.HANN,
#                        slide_length=512, normal_type=SpectralFilterBankNormalType.AREA,
#                        is_scale=True); spec = obj.cqt(x)
obj = af.CQT(num=84, samplate=sr, low_fre=32.703, bin_per_octave=12,
             factor=1.0, beta=0.0, thresh=0.01, window_type=WindowType.HANN,
             slide_length=512, normal_type=SpectralFilterBankNormalType.AREA,
             is_scale=True)
spec = obj.cqt(x)
save("cqt", "cqt_area", spec=spec, freq=obj.get_fre_band_arr(),
     fft_length=obj.get_fft_length(), time_len=obj.cal_time_length(len(x)),
     slide_length=512)

# cqt_none: same with normal_type=NONE
obj = af.CQT(num=84, samplate=sr, low_fre=32.703, bin_per_octave=12,
             factor=1.0, beta=0.0, thresh=0.01, window_type=WindowType.HANN,
             slide_length=512, normal_type=SpectralFilterBankNormalType.NONE,
             is_scale=True)
spec = obj.cqt(x)
save("cqt", "cqt_none", spec=spec, freq=obj.get_fre_band_arr(),
     fft_length=obj.get_fft_length(), slide_length=512)

# vqt: beta=20 (variable-Q), 24 bins per octave over 6 octaves from C2
obj = af.CQT(num=144, samplate=sr, low_fre=65.406, bin_per_octave=24,
             factor=1.0, beta=20.0, thresh=0.01, window_type=WindowType.HANN,
             slide_length=512, normal_type=SpectralFilterBankNormalType.AREA,
             is_scale=True)
spec = obj.cqt(x)
save("cqt", "vqt", spec=spec, freq=obj.get_fre_band_arr(),
     fft_length=obj.get_fft_length(), slide_length=512)

# cqt_noise / vqt_noise: the same calls on seeded white noise (every octave carries
# energy, so the top octave can be compared tightly); the signal is saved too
rng = np.random.RandomState(0)
noise = np.ascontiguousarray(0.1 * rng.randn(len(x)), dtype=np.float32)
obj = af.CQT(num=84, samplate=sr, low_fre=32.703, bin_per_octave=12,
             factor=1.0, beta=0.0, thresh=0.01, window_type=WindowType.HANN,
             slide_length=512, normal_type=SpectralFilterBankNormalType.AREA,
             is_scale=True)
spec = obj.cqt(noise)
save("cqt", "cqt_noise", spec=spec, x=noise, freq=obj.get_fre_band_arr(), slide_length=512)
obj = af.CQT(num=144, samplate=sr, low_fre=65.406, bin_per_octave=24,
             factor=1.0, beta=20.0, thresh=0.01, window_type=WindowType.HANN,
             slide_length=512, normal_type=SpectralFilterBankNormalType.AREA,
             is_scale=True)
spec = obj.cqt(noise)
save("cqt", "vqt_noise", spec=spec, x=noise, freq=obj.get_fre_band_arr(), slide_length=512)

# cqt_chroma: obj.chroma(spec, chroma_num=12, data_type=POWER, norm_type=MAX)
from audioflux.type import SpectralDataType, ChromaDataNormalType
obj = af.CQT(num=84, samplate=sr, low_fre=32.703, bin_per_octave=12,
             slide_length=512, normal_type=SpectralFilterBankNormalType.AREA)
spec = obj.cqt(x)
chroma = obj.chroma(spec, chroma_num=12, data_type=SpectralDataType.POWER,
                    norm_type=ChromaDataNormalType.MAX)
save("cqt", "cqt_chroma", chroma=chroma, spec=spec)
