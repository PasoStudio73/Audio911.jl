"""Shared helpers of the audioFlux oracle scripts.

Every script in this directory generates reference outputs of audioFlux
(https://github.com/libAudioFlux/audioFlux, MIT licence) for
test/test_files/test.wav and saves them as .mat files under
test/audioflux_files/<feature>/. Run them with `test/run.sh oracle`.

The wav is read as float32 samples in [-1, 1) exactly like Audio911's loader
(PCM16 / 32768), so both sides analyse the same signal.
"""
import os
import numpy as np
import scipy.io
import scipy.io.wavfile

TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAV = os.path.join(TEST_DIR, "test_files", "test.wav")
OUT = os.path.join(TEST_DIR, "audioflux_files")


def load(n=None):
    """Mono float32 signal and sample rate; the first `n` samples when given."""
    sr, x = scipy.io.wavfile.read(WAV)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    else:
        x = x.astype(np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32)
    if n is not None:
        x = x[:n]
    return x, int(sr)


def save(feature, name, **arrays):
    """Write arrays to test/audioflux_files/<feature>/<name>.mat."""
    d = os.path.join(OUT, feature)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name + ".mat")
    scipy.io.savemat(path, {k: np.asarray(v) for k, v in arrays.items()}, do_compression=True)
    print("  wrote", os.path.relpath(path, TEST_DIR))
