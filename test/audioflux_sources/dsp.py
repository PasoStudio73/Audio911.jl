"""audioFlux DSP and utility functions (test/audioflux_files/dsp/).

Julia counterpart: test/dsp_af.jl. CZT, Xcorr, conv (C only, through
ctypes), the A/B/C/D weightings, the dB conversions, the feature scalers,
temproal_db and synth_f0.
"""
import numpy as np
from ctypes import c_int, c_void_p, POINTER, pointer, byref
import audioflux as af
from audioflux.type import XcorrNormalType, WindowType
from audioflux.fftlib import get_fft_lib
from _common import load, save

x, sr = load()
out = {"sr": sr}
seg = np.ascontiguousarray(x[4000:4000 + 1024])
# obj = af.CZT(radix2_exp=10); X = obj.czt(seg, low_w, high_w)
# The wrapper passes 1024-sample buffers but cztObj_czt reads 2048 (the FFT
# length), so its output is NaN from the garbage past the end; the C function
# is called here with zero-padded 2048-sample buffers (first 1024 values kept).
from ctypes import c_float
czt = af.CZT(radix2_exp=10)
fn = czt._lib["cztObj_czt"]
f1 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
fn.argtypes = [type(czt._obj), f1, f1, c_float, c_float, f1, f1]
re = np.zeros(2048, dtype=np.float32); re[:1024] = seg
im = np.zeros(2048, dtype=np.float32)
for name, lo, hi in (("czt_full", 0.0, 1.0), ("czt_band", 0.05, 0.2)):
    o_re = np.zeros(2048, dtype=np.float32); o_im = np.zeros(2048, dtype=np.float32)
    fn(czt._obj, re, im, c_float(lo), c_float(hi), o_re, o_im)
    out[name] = (o_re + 1j * o_im)[:1024]
out["czt_x"] = seg

# obj = af.Xcorr(); r, m = obj.xcorr(a, b, xcorr_normal_type)
a = np.ascontiguousarray(x[8000:10048]); b = np.ascontiguousarray(x[8100:10148])
out["xc_a"], out["xc_b"] = a, b
out["xc_none"], _ = af.Xcorr().xcorr(a, b, XcorrNormalType.NONE)
out["xc_coeff"], _ = af.Xcorr().xcorr(a, b, XcorrNormalType.COEFF)
out["xc_auto"], _ = af.Xcorr().xcorr(a, None, XcorrNormalType.COEFF)

# convObj_conv(obj, a, n, k, m, &mode, &method, out)   (C only)
lib = get_fft_lib()
f1 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
new = lib["convObj_new"]; new.argtypes = [POINTER(c_void_p)]
conv = lib["convObj_conv"]
conv.argtypes = [c_void_p, f1, c_int, f1, c_int, POINTER(c_int), POINTER(c_int), f1]
conv.restype = c_int
sig = np.ascontiguousarray(x[12000:12500])
for kn in (31, 32):
    k = np.ascontiguousarray(np.hanning(kn).astype(np.float32))
    out[f"conv_k{kn}"] = k
    for mode in (0, 1, 2):
        for method in (1, 2):                     # direct, fft
            o = c_void_p(); new(byref(o))
            buf = np.zeros(len(sig) + kn, dtype=np.float32)
            n = conv(o, sig, len(sig), k, kn, pointer(c_int(mode)), pointer(c_int(method)), buf)
            out[f"conv_{kn}_{mode}_{method}"] = buf[:n]
out["conv_sig"] = sig

# af.utils.auditory_weight_{a,b,c,d}(freqs)
f = np.linspace(10, 20000, 400).astype(np.float32)
out["w_f"] = f
for w in "abcd":
    out["w_" + w] = getattr(af.utils, "auditory_weight_" + w)(f)

# dB conversions of a magnitude STFT
S = np.abs(af.STFT(radix2_exp=10, window_type=WindowType.HANN, slide_length=256).stft(x)).astype(np.float32)
P = S ** 2
out["S"] = S
out["db_rel"] = af.utils.power_to_db(P, min_db=-80)
out["db_abs"] = af.utils.power_to_abs_db(P, fft_length=1024, is_norm=False, min_db=-80)
out["db_abs_norm"] = af.utils.power_to_abs_db(P, fft_length=1024, is_norm=True, min_db=-80)
out["db_mag"] = af.utils.mag_to_abs_db(S, fft_length=1024, is_norm=False, min_db=-80)
out["log_c"] = af.utils.log_compress(S, gamma=10.0)
out["log10_c"] = af.utils.log10_compress(S, gamma=10.0)

# feature scalers on a samples x features matrix, and on sorted columns
rng = np.random.default_rng(0)
M = rng.standard_normal((101, 5)).astype(np.float32) * np.array([1, 2, 3, 4, 5], dtype=np.float32)
Ms = np.ascontiguousarray(np.sort(M, axis=0))
out["sc_M"], out["sc_Ms"] = M, Ms
for name in ("min_max", "max_abs", "robust", "center", "mean", "arctan"):
    out["sc_" + name] = getattr(af.utils, name + "_scale")(M)
    out["scs_" + name] = getattr(af.utils, name + "_scale")(Ms)
out["sc_stand_0"] = af.utils.stand_scale(M, tp=0)
out["sc_stand_1"] = af.utils.stand_scale(M, tp=1)

# temproal_db(x, base) -> (max, avg, percent)
out["tdb"] = np.array(af.utils.temproal_db(x, base=18.0), dtype=np.float64)
# synth_f0(times, f0, sr, amplitudes)
times = np.arange(0, 50) * (256 / 16000)
f0 = 220 + 100 * np.sin(np.arange(50) / 7)
amp = np.linspace(0.2, 0.8, 50)
out["sf_times"], out["sf_f0"], out["sf_amp"] = times, f0, amp
out["sf_y"] = af.utils.synth_f0(times, f0, 16000, amp)
out["sf_y1"] = af.utils.synth_f0(times, f0, 16000)
save("dsp", "dsp", **out)
