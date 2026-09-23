"""audioFlux BFT filterbank spectrograms: every scale and filter style
(test/audioflux_files/fbank/).

Julia counterpart: test/af_fbank.jl. A BFT with a scale is the STFT power
spectrogram (Hann window of 512, hop 256) multiplied by the filterbank, so
the .mat files check Audio911's `MelSpec(Stft(...); scale, style, norm)`.
"""
import numpy as np
import audioflux as af
from audioflux.type import (WindowType, SpectralDataType, SpectralFilterBankScaleType,
                            SpectralFilterBankStyleType, SpectralFilterBankNormalType)
from _common import load, save

x, sr = load()
R2 = 9  # fft_length 512

cases = {
    # name: (num, low, high, bin_per_octave, scale, style, norm)
    "mel_slaney_none":     (26, 0.0, 8000.0, 12, "MEL", "SLANEY", "NONE"),
    "mel_slaney_area":     (26, 0.0, 8000.0, 12, "MEL", "SLANEY", "AREA"),
    "mel_slaney_bw":       (26, 0.0, 8000.0, 12, "MEL", "SLANEY", "BAND_WIDTH"),
    "bark_slaney_none":    (20, 0.0, 8000.0, 12, "BARK", "SLANEY", "NONE"),
    "erb_slaney_none":     (20, 0.0, 8000.0, 12, "ERB", "SLANEY", "NONE"),
    "linspace_slaney_none": (20, 100.0, 7000.0, 12, "LINSPACE", "SLANEY", "NONE"),
    "octave_slaney_none":  (84, 32.703, 8000.0, 12, "OCTAVE", "SLANEY", "NONE"),
    "log_slaney_none":     (40, 100.0, 7000.0, 12, "LOG", "SLANEY", "NONE"),
    "mel_hann_none":       (26, 0.0, 8000.0, 12, "MEL", "HANN", "NONE"),
    "mel_hamm_area":       (26, 0.0, 8000.0, 12, "MEL", "HAMM", "AREA"),
    "mel_blackman_none":   (26, 0.0, 8000.0, 12, "MEL", "BLACKMAN", "NONE"),
    "mel_bohman_none":     (26, 0.0, 8000.0, 12, "MEL", "BOHMAN", "NONE"),
    "mel_kaiser_none":     (26, 0.0, 8000.0, 12, "MEL", "KAISER", "NONE"),
    "mel_gauss_none":      (26, 0.0, 8000.0, 12, "MEL", "GAUSS", "NONE"),
    "mel_rect_none":       (26, 0.0, 8000.0, 12, "MEL", "RECT", "NONE"),
    "mel_point_none":      (26, 0.0, 8000.0, 12, "MEL", "POINT", "NONE"),
    "mel_etsi_none":       (26, 0.0, 8000.0, 12, "MEL", "ETSI", "NONE"),
}

for name, (num, low, high, bpo, scale, style, norm) in cases.items():
    # obj = af.BFT(num=num, radix2_exp=9, samplate=sr, low_fre=low, high_fre=high,
    #              bin_per_octave=bpo, window_type=WindowType.HANN, slide_length=256,
    #              scale_type=SpectralFilterBankScaleType.<scale>,
    #              style_type=SpectralFilterBankStyleType.<style>,
    #              normal_type=SpectralFilterBankNormalType.<norm>,
    #              data_type=SpectralDataType.POWER); spec = obj.bft(x, result_type=1)
    obj = af.BFT(num=num, radix2_exp=R2, samplate=sr, low_fre=low, high_fre=high,
                 bin_per_octave=bpo, window_type=WindowType.HANN, slide_length=256,
                 scale_type=getattr(SpectralFilterBankScaleType, scale),
                 style_type=getattr(SpectralFilterBankStyleType, style),
                 normal_type=getattr(SpectralFilterBankNormalType, norm),
                 data_type=SpectralDataType.POWER)
    spec = obj.bft(x, result_type=1)
    spec = np.real(spec)
    save("fbank", name, spec=spec, freq=obj.get_fre_band_arr(),
         time_len=obj.cal_time_length(len(x)), num=num, low=low, high=high, bpo=bpo)

# the plain linear BFT is the STFT power spectrogram itself
obj = af.BFT(num=257, radix2_exp=R2, samplate=sr, low_fre=0.0, high_fre=8000.0,
             window_type=WindowType.HANN, slide_length=256,
             scale_type=SpectralFilterBankScaleType.LINEAR,
             style_type=SpectralFilterBankStyleType.SLANEY,
             data_type=SpectralDataType.POWER)
spec = np.real(obj.bft(x, result_type=1))
save("fbank", "linear", spec=spec, time_len=obj.cal_time_length(len(x)))
