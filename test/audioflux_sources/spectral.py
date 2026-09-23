"""audioFlux spectral descriptors (test/audioflux_files/spectral/).

Julia counterpart: test/spectral_af.jl. Every feature of af.Spectral runs on
the magnitude STFT of test.wav (periodic Hann window of 512, hop 256), the
phase-based ones with its phase.
"""
import numpy as np
import audioflux as af
from audioflux.type import WindowType, SpectralNoveltyMethodType as NM, SpectralNoveltyDataType as ND
from _common import load, save

x, sr = load()
stft = af.STFT(radix2_exp=9, window_type=WindowType.HANN, slide_length=256)
S = stft.stft(x)                       # (257, time) complex
mag = np.ascontiguousarray(np.abs(S), dtype=np.float32)
phase = np.ascontiguousarray(np.angle(S), dtype=np.float32)
freq = np.linspace(0, sr / 2, 257, dtype=np.float32)

# obj = af.Spectral(num=257, fre_band_arr=freq); obj.set_time_length(T); obj.<feature>(mag, ...)
obj = af.Spectral(num=257, fre_band_arr=freq)
obj.set_time_length(mag.shape[1])
out = {"mag": mag, "phase": phase}
out["flatness"] = obj.flatness(mag)
out["flux"] = obj.flux(mag)                                   # step=1, p=2, no root, sum
out["flux_p1_pos"] = obj.flux(mag, step=2, p=1, is_positive=True, is_exp=False, tp=1)
out["flux_exp"] = obj.flux(mag, step=1, p=2, is_positive=False, is_exp=True, tp=0)
out["rolloff"] = obj.rolloff(mag, threshold=0.9)
out["centroid"] = obj.centroid(mag)
out["spread"] = obj.spread(mag)
out["skewness"] = obj.skewness(mag)
out["kurtosis"] = obj.kurtosis(mag)
out["entropy"] = obj.entropy(mag, is_norm=False)
out["entropy_norm"] = obj.entropy(mag, is_norm=True)
out["crest"] = obj.crest(mag)
out["slope"] = obj.slope(mag)
out["decrease"] = obj.decrease(mag)
out["band_width"] = obj.band_width(mag, p=2)
out["band_width_3"] = obj.band_width(mag, p=3)
out["rms"] = obj.rms(mag)
out["energy"] = obj.energy(mag)
out["energy_log"] = obj.energy(mag, is_log=True, gamma=10.0)
out["hfc"] = obj.hfc(mag)
out["sd"] = obj.sd(mag)
out["sd_pos"] = obj.sd(mag, step=2, is_positive=True)
out["sf"] = obj.sf(mag)
out["mkl"] = obj.mkl(mag)
out["mkl_mean"] = obj.mkl(mag, tp=1)
out["pd"] = obj.pd(mag, phase)
out["wpd"] = obj.wpd(mag, phase)
out["nwpd"] = obj.nwpd(mag, phase)
out["cd"] = obj.cd(mag, phase)
out["rcd"] = obj.rcd(mag, phase)
out["broadband"] = obj.broadband(mag, threshold=1)   # the wrapper accepts 0 or 1
out["novelty"] = obj.novelty(mag)
out["novelty_kl_number"] = obj.novelty(mag, step=1, threshold=0.001, method_type=NM.KL, data_type=ND.NUMBER)
out["novelty_entropy"] = obj.novelty(mag, step=2, threshold=0.0, method_type=NM.ENTROY, data_type=ND.VALUE)
out["novelty_is"] = obj.novelty(mag, step=1, threshold=0.0, method_type=NM.IS, data_type=ND.VALUE)
out["eef"] = obj.eef(mag)
out["eef_norm"] = obj.eef(mag, is_norm=True)
out["eer"] = obj.eer(mag, gamma=10.0)
v, f = obj.max(mag); out["max"] = v; out["max_fre"] = f
v, f = obj.mean(mag); out["mean"] = v; out["mean_fre"] = f
v, f = obj.var(mag); out["var"] = v; out["var_fre"] = f
save("spectral", "spectral", **out)
