"""audioFlux Viterbi and discrete HMM (test/audioflux_files/hmm/).

Julia counterpart: test/hmm_af.jl. Neither is wrapped in Python, so the C
functions are called through ctypes. The model parameters are dyadic so
that their float32 sums are exactly 1 (hmmObj_init compares them with ==).
The observations are sampled with numpy from the same model.
"""
import numpy as np
from ctypes import c_int, c_float, c_void_p, POINTER, pointer, byref
from audioflux.fftlib import get_fft_lib
from _common import save

lib = get_fft_lib()
f32 = lambda nd: np.ctypeslib.ndpointer(dtype=np.float32, ndim=nd, flags="C_CONTIGUOUS")
i32 = lambda nd: np.ctypeslib.ndpointer(dtype=np.int32, ndim=nd, flags="C_CONTIGUOUS")

S, K, T = 3, 4, 40
pi = np.array([0.5, 0.25, 0.25], dtype=np.float32)
A = np.array([[0.75, 0.125, 0.125], [0.25, 0.5, 0.25], [0.125, 0.125, 0.75]], dtype=np.float32)
B = np.array([[0.5, 0.25, 0.125, 0.125], [0.125, 0.5, 0.25, 0.125], [0.125, 0.125, 0.25, 0.5]], dtype=np.float32)
rng = np.random.default_rng(0)
s = [rng.choice(S, p=pi)]
for t in range(1, T):
    s.append(rng.choice(S, p=A[s[-1]]))
obs = np.array([rng.choice(K, p=B[q]) for q in s], dtype=np.int32)

# prob = viterbi(pi, A, B, S, K, obs, T, &isLog, sArr, mProbArr, mIndexArr)
vit = lib["viterbi"]
vit.argtypes = [f32(1), f32(2), f32(2), c_int, c_int, i32(1), c_int, POINTER(c_int), i32(1), f32(2), i32(2)]
vit.restype = c_float
out = {"pi": pi, "A": A, "B": B, "obs": obs + 1}
for is_log in (0, 1):
    sArr = np.zeros(T, dtype=np.int32)
    prob = np.zeros((T, S), dtype=np.float32)
    idx = np.zeros((T, S), dtype=np.int32)
    p = vit(pi, A, B, S, K, obs, T, pointer(c_int(is_log)), sArr, prob, idx)
    # backtrack audioFlux's own pointers into the Viterbi path
    path = [int(np.argmax(prob[-1]))]
    for t in range(T - 1, 0, -1):
        path.append(int(idx[t, path[-1]]))
    out[f"vit_prob_{is_log}"] = p
    out[f"vit_argmax_{is_log}"] = sArr + 1
    out[f"vit_path_{is_log}"] = np.array(path[::-1]) + 1
    out[f"vit_scores_{is_log}"] = prob

# hmmObj_new / hmmObj_init / hmmObj_predict / hmmObj_decode / hmmObj_train
new = lib["hmmObj_new"]; new.argtypes = [POINTER(c_void_p), c_int, c_int]
init = lib["hmmObj_init"]; init.argtypes = [c_void_p, f32(1), f32(2), f32(2)]
pred = lib["hmmObj_predict"]; pred.argtypes = [c_void_p, i32(1), c_int]; pred.restype = c_float
dec = lib["hmmObj_decode"]; dec.argtypes = [c_void_p, i32(1), c_int, i32(1), f32(2)]; dec.restype = c_float
train = lib["hmmObj_train"]; train.argtypes = [c_void_p, i32(1), c_int, POINTER(c_int), POINTER(c_float)]
h = c_void_p()
new(byref(h), S, K)
# the object keeps these arrays (and train overwrites them), so they are copies
hp, hA, hB = pi.copy(), A.copy(), B.copy()
init(h, hp, hA, hB)
out["hmm_predict"] = pred(h, obs, T)
sArr = np.zeros(T, dtype=np.int32); prob = np.zeros((T, S), dtype=np.float32)
out["hmm_decode"] = dec(h, obs, T, sArr, prob)
# train from a flatter start
tp = np.array([0.25, 0.5, 0.25], dtype=np.float32)
tA = np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]], dtype=np.float32)
tB = np.array([[0.25, 0.25, 0.25, 0.25], [0.5, 0.25, 0.125, 0.125], [0.125, 0.125, 0.25, 0.5]], dtype=np.float32)
out["train_pi0"], out["train_A0"], out["train_B0"] = tp.copy(), tA.copy(), tB.copy()
for iters in (1, 5, 100):
    p0, A0, B0 = out["train_pi0"].copy(), out["train_A0"].copy(), out["train_B0"].copy()
    h3 = c_void_p(); new(byref(h3), S, K); init(h3, p0, A0, B0)
    train(h3, obs, T, pointer(c_int(iters)), pointer(c_float(1e-3)))
    out[f"train_pi_{iters}"], out[f"train_A_{iters}"], out[f"train_B_{iters}"] = p0, A0, B0
# (the objects are not freed: hmmObj_free would free the numpy buffers)
save("hmm", "hmm", **out)
