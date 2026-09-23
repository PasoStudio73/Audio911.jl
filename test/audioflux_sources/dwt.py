"""audioFlux DWT, WPT and SWT reference outputs (test/audioflux_files/dwt/).

Julia counterpart: test/dwt.jl. First 2^12 samples of test.wav.
"""
import re
import numpy as np
import audioflux as af
from audioflux.type import WaveletDiscreteType as WD
from _common import load, save

x, sr = load(2 ** 12)

NAMES = ["haar"] + [f"db{n}" for n in (2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40)] \
    + [f"sym{n}" for n in (2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30)] \
    + [f"coif{n}" for n in (1, 2, 3, 4, 5)] + [f"fk{n}" for n in (4, 6, 8, 14, 18, 22)] \
    + ["bior1.1", "bior1.3", "bior1.5", "bior2.2", "bior2.4", "bior2.6", "bior2.8",
       "bior3.1", "bior3.3", "bior3.5", "bior3.7", "bior3.9", "bior4.4", "bior5.5", "bior6.8", "dmey"]


def wtype(name):
    """audioFlux (wavelet_type, t1, t2) of a wavelet name."""
    if name == "haar":
        return WD.HAAR, 0, 0
    if name == "dmey":
        return WD.DMEY, 0, 0
    m = re.fullmatch(r"bior(\d)\.(\d)", name)
    if m:
        return WD.BIOR, int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"([a-z]+)(\d+)", name)
    return {"db": WD.DB, "sym": WD.SYM, "coif": WD.COIF, "fk": WD.FK}[m.group(1)], int(m.group(2)), 0


# audioFlux 0.1.9's Python DWT passes an extra `samplate` pointer to the C
# function dwtObj_new(obj, num, radix2Exp, *waveletType, *t1, *t2), so the
# wavelet never reaches the C code and every DWT is sym4. The DWT fixtures
# therefore call the C library of the same package directly with the
# declared signature (the WPT and SWT wrappers are correct and are used).
from ctypes import Structure, POINTER, pointer, c_int
from audioflux.fftlib import get_fft_lib


class OpaqueDWT(Structure):
    _fields_ = []


def c_dwt(x, num, radix2_exp, wt, t1, t2):
    lib = get_fft_lib()
    obj = pointer(OpaqueDWT())
    new = lib["dwtObj_new"]
    new.argtypes = [POINTER(POINTER(OpaqueDWT)), c_int, c_int, POINTER(c_int), POINTER(c_int), POINTER(c_int)]
    new(obj, c_int(num), c_int(radix2_exp), pointer(c_int(wt.value)), pointer(c_int(t1)), pointer(c_int(t2)))
    fn = lib["dwtObj_dwt"]
    fn.argtypes = [POINTER(OpaqueDWT),
                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")]
    n = 1 << radix2_exp
    coef = np.zeros(n, dtype=np.float32)
    m = np.zeros((num, n), dtype=np.float32)
    fn(obj, np.ascontiguousarray(x[:n], dtype=np.float32), coef, m)
    free = lib["dwtObj_free"]
    free.argtypes = [POINTER(OpaqueDWT)]
    free(obj)
    return coef, m


# dwt_<name>: dwtObj_new(&obj, 11, 12, &wavelet_type, &t1, &t2); dwtObj_dwt(obj, x, coef, m)
for name in NAMES:
    wt, t1, t2 = wtype(name)
    coef, m = c_dwt(x, 11, 12, wt, t1, t2)
    save("dwt", "dwt_" + name.replace(".", "_"), coef=coef, image=m)

# dwt_sym4_level5: a partial decomposition
coef, m = c_dwt(x, 5, 12, WD.SYM, 4, 0)
save("dwt", "dwt_sym4_level5", coef=coef, image=m)
# the Python wrapper's result, for the record: always sym4
obj = af.DWT(num=11, radix2_exp=12, samplate=sr, wavelet_type=WD.DB, t1=2, t2=0)
coef, m = obj.dwt(x)
save("dwt", "dwt_wrapper_db2", coef=coef, image=m)

# wpt_<name>_<level>: af.WPT(num=level, radix2_exp=12, samplate=sr, wavelet_type=..., t1, t2)
for name, level in [("sym4", 5), ("db4", 3), ("bior3.5", 4), ("haar", 6)]:
    wt, t1, t2 = wtype(name)
    obj = af.WPT(num=level, radix2_exp=12, samplate=sr, wavelet_type=wt, t1=t1, t2=t2)
    coef, m = obj.wpt(x)
    save("dwt", f"wpt_{name.replace('.', '_')}_{level}", coef=coef, image=m)

# swt_<name>_<level>: af.SWT(num=level, fft_length=4096, wavelet_type=..., t1, t2); a, d = obj.swt(x)
for name, level in [("sym4", 5), ("db2", 3), ("haar", 4), ("coif3", 2)]:
    wt, t1, t2 = wtype(name)
    obj = af.SWT(num=level, fft_length=4096, wavelet_type=wt, t1=t1, t2=t2)
    a, d = obj.swt(x)
    save("dwt", f"swt_{name}_{level}", app=a, det=d)
