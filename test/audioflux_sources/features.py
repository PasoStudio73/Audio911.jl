"""audioFlux xxcc, deconv, cepstrogram, temporal, harmonic ratio and linear
chroma (test/audioflux_files/features/).

Julia counterpart: test/af_features.jl.
"""
import numpy as np
import audioflux as af
from audioflux.type import WindowType, CepstralRectifyType, CepstralEnergyType
from _common import load, save

x, sr = load()
stft = af.STFT(radix2_exp=9, window_type=WindowType.HANN, slide_length=256)
S = stft.stft(x)
mag = np.ascontiguousarray(np.abs(S), dtype=np.float32)
power = np.ascontiguousarray(mag ** 2, dtype=np.float32)
out = {}

# xxcc: af.XXCC(num=257); obj.set_time_length(T); obj.xxcc(mag, cc_num=13, rectify_type=LOG|CUBIC_ROOT)
obj = af.XXCC(num=257)
obj.set_time_length(mag.shape[1])
out["xxcc_log"] = obj.xxcc(mag, cc_num=13, rectify_type=CepstralRectifyType.LOG)
out["xxcc_cbrt"] = obj.xxcc(mag, cc_num=13, rectify_type=CepstralRectifyType.CUBIC_ROOT)
# xxcc_standard with energy = sum of the power spectrum, replace and append
energy = np.ascontiguousarray(power.sum(axis=0), dtype=np.float32)
c, d1, d2 = obj.xxcc_standard(mag, energy, cc_num=13, delta_window_length=9,
                              energy_type=CepstralEnergyType.REPLACE, rectify_type=CepstralRectifyType.LOG)
out["std_replace"], out["std_replace_d1"], out["std_replace_d2"] = c, d1, d2
# APPEND through the C function: the 0.1.9 wrapper passes cc_num + 1 to C,
# which adds the energy row again and writes past its buffers
from ctypes import POINTER, pointer, c_int
fn = obj._lib["xxccObj_xxccStandard"]
f2 = lambda: np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
fn.argtypes = [type(obj._obj), f2(), c_int, np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
               POINTER(c_int), POINTER(c_int), POINTER(c_int), f2(), f2(), f2()]
T = mag.shape[1]
c = np.zeros((T, 14), dtype=np.float32); d1 = np.zeros_like(c); d2 = np.zeros_like(c)
fn(obj._obj, np.ascontiguousarray(mag.T), c_int(13), energy, pointer(c_int(9)),
   pointer(c_int(CepstralEnergyType.APPEND.value)), pointer(c_int(CepstralRectifyType.LOG.value)), c, d1, d2)
out["std_append"], out["std_append_d1"], out["std_append_d2"] = c.T, d1.T, d2.T
out["energy"] = energy

# deconv: af.Deconv(num=257); obj.set_time_length(T); timbre, pitch = obj.deconv(mag)
obj = af.Deconv(num=257)
obj.set_time_length(mag.shape[1])
timbre, pitch = obj.deconv(mag)
out["deconv_timbre"], out["deconv_pitch"] = timbre, pitch

# cepstrogram: af.Cepstrogram(radix2_exp=9, samplate=sr, window_type=RECT, slide_length=256)
#              ceps, env, det = obj.cepstrogram(x, cep_num=8)
obj = af.Cepstrogram(radix2_exp=9, samplate=sr, window_type=WindowType.RECT, slide_length=256)
ceps, env, det = obj.cepstrogram(x, cep_num=8)
out["cepstrum"], out["envelope"], out["details"] = ceps, env, det

# temporal: af.Temporal(frame_length=512, slide_length=256, window_type=HANN)
#           d = obj.temporal(x, has_energy=True, has_rms=True, has_zcr=True)
#           ezr: temporalObj_ezr(obj, 10.0, out) through ctypes (not wrapped in Python)
obj = af.Temporal(frame_length=512, slide_length=256, window_type=WindowType.HANN)
d = obj.temporal(x, has_energy=True, has_rms=True, has_zcr=True)
out["t_energy"], out["t_rms"], out["t_zcr"] = d["energy_arr"], d["rms_arr"], d["zcr_arr"]
from ctypes import POINTER, c_float
fn = obj._lib["temporalObj_ezr"]
fn.argtypes = [type(obj._obj), c_float, np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")]
ezr = np.zeros(len(d["energy_arr"]), dtype=np.float32)
fn(obj._obj, c_float(10.0), ezr)
out["t_ezr"] = ezr

# harmonic ratio: af.HarmonicRatio(samplate=sr, low_fre=32.703, radix2_exp=12, window_type=HAMM, slide_length=1024)
obj = af.HarmonicRatio(samplate=sr, low_fre=32.703, radix2_exp=12, window_type=WindowType.HAMM, slide_length=1024)
out["hr"] = obj.harmonic_ratio(x)
obj = af.HarmonicRatio(samplate=sr, low_fre=50.0, radix2_exp=10, window_type=WindowType.HAMM, slide_length=256)
out["hr_small"] = obj.harmonic_ratio(x)

# chroma_linear: af.chroma_linear(x, chroma_num=12, radix2_exp=9, samplate=sr, low_fre=0, high_fre=8000,
#                                  window_type=HANN, slide_length=256)
try:
    out["chroma_linear"] = af.chroma_linear(x, chroma_num=12, radix2_exp=9, samplate=sr, low_fre=0.0,
                                            high_fre=8000.0, window_type=WindowType.HANN, slide_length=256)
except TypeError as err:
    print("  chroma_linear:", err)
save("features", "features", **out)
