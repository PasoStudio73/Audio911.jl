```@meta
CurrentModule = Audio911
```

# [MFCC variants](@id mfcc_variants)

"MFCC" names a family, not one algorithm. Every reference implementation
makes its own choices along the same handful of axes, and two toolkits fed
the same audio disagree by a lot. Audio911 exposes every axis as a keyword
of the stage that owns it, and ships one preset per reference implementation
so that a published recipe is a single call. A preset is nothing more than a
composition of the public stages; copy its body and change any knob.

## The axes

| axis | stage / keyword | choices |
|:-----|:----------------|:--------|
| pre-emphasis | `Frames(preemph=)` per frame, or [`preemphasis`](@ref) on the signal | none, 0.97 per frame (HTK, Kaldi, ETSI), 0.97 on the signal (python_speech_features, librosa) |
| DC removal | `Frames(dc_removal=)`, [`offset_compensation`](@ref) | none, per-frame mean (Kaldi), notch filter (ETSI) |
| framing | `Frames(winsize, winstep, center, pad_end)` | snip edges (MATLAB, HTK, Kaldi, ETSI), centred + padded (librosa), last frame padded (python_speech_features) |
| window | `Frames(type, periodic)` | periodic Hamming (MATLAB), symmetric Hamming (HTK, ETSI), Povey (Kaldi), periodic Hann (librosa), rectangular (python_speech_features) |
| FFT size | `Stft(nfft)` | window length (MATLAB), next power of two (HTK, Kaldi, ETSI), 512 (python_speech_features), 2048 (librosa) |
| spectrum | `Stft(spectrum, scale)` | power (MATLAB, Kaldi, librosa), magnitude (HTK, ETSI), power / nfft (python_speech_features) |
| window normalisation | `MelSpec(win_norm)` | on (MATLAB), off (everyone else) |
| mel scale | `auditory_fbank(scale)` | HTK formula `2595 log10(1 + f/700)` (MATLAB, HTK, Kaldi, ETSI, python_speech_features), Slaney (librosa) |
| filter shape | `auditory_fbank(domain)`, [`etsi_fbank`](@ref), [`psf_fbank`](@ref) | triangles in Hz (MATLAB, librosa), triangles in mel (HTK, Kaldi), integer-bin triangles (ETSI, python_speech_features) |
| filter normalisation | `auditory_fbank(norm)` | bandwidth = `2/bw` (MATLAB, librosa), none (HTK, Kaldi, ETSI, python_speech_features) |
| bands, range | `nbands`, `freqrange` | 32 (MATLAB), 20 (HTK), 23 from 20 Hz (Kaldi), 128 (librosa), 23 from 64 Hz (ETSI), 26 (python_speech_features) |
| floor before the log | `Mfcc(floor)` | `floatmin` (MATLAB), 1.0 on 16-bit samples (HTK), `eps(Float32)` (Kaldi), `1e-10` (librosa), `2e-22` (ETSI), `eps` (python_speech_features) |
| rectification | `Mfcc(rect, top_db)` | `log10` or cubic root (MATLAB), natural log (HTK, Kaldi, ETSI, python_speech_features), `10 log10` clipped to 80 dB (librosa) |
| DCT scaling | `Mfcc(dct)` | orthonormal (MATLAB, Kaldi, librosa, python_speech_features), `sqrt(2/N)` on every row (HTK), none (ETSI) |
| coefficients | `Mfcc(ncoeffs, first)` | 13 from C0 (MATLAB, Kaldi, ETSI, python_speech_features), 12 from C1 (HTK), 20 (librosa) |
| liftering | `Mfcc(lifter, lifter_offset)` | none (MATLAB, ETSI), 22 (HTK, Kaldi, python_speech_features), off by default and index shifted by one when on (librosa) |
| energy | `Mfcc(energy, energy_mode, energy_floor)` | none (MATLAB, librosa), raw log energy replacing C0 (Kaldi), spectrum log energy replacing C0 (python_speech_features), raw log energy appended (HTK `_E`, ETSI) |
| deltas | `Delta(delta_length)` | regression window 9 (MATLAB), 5 = `DELTAWINDOW 2` (HTK), 5 = `delta-window 2` (Kaldi), Savitzky–Golay width 9 (librosa) |

## Presets

Each preset returns an [`Mfcc`](@ref) whose parents can be walked back
(`get_parent`, `get_frontend`, `get_frames`) to inspect the intermediate
stages.

```@docs
mfcc_matlab
mfcc_htk
mfcc_kaldi
mfcc_librosa
mfcc_etsi
mfcc_psf
etsi_fbank
psf_fbank
offset_compensation
```

### References

- MATLAB Audio Toolbox, `audioFeatureExtractor` and `mfcc` documentation
  (R2023b). The MATLAB preset is the chain verified against the fixtures in
  `test/matlab_files/mfcc`.
- S. Young et al., *The HTK Book* (v3.4), §5.6 "Filterbank analysis",
  §5.8 "Cepstral features", §5.9 "Energy measures"; `HSigP.c`
  (`Wave2FBank`, `FBank2MFCC`, `WeightCepstrum`).
- D. Povey et al., "The Kaldi speech recognition toolkit", ASRU 2011;
  `feat/feature-mfcc.cc`, `feat/feature-window.cc`, `feat/mel-computations.cc`
  (defaults of `MfccOptions`, `FrameExtractionOptions`, `MelBanksOptions`).
- B. McFee et al., "librosa: audio and music signal analysis in Python",
  SciPy 2015; `librosa.feature.mfcc`, `librosa.feature.melspectrogram`,
  `librosa.filters.mel`, `librosa.power_to_db` (0.10).
- ETSI ES 201 108 v1.1.3 (2003), "Speech Processing, Transmission and
  Quality Aspects (STQ); Distributed speech recognition; Front-end feature
  extraction algorithm; Compression algorithms", §4.2.
- J. Lyons et al., `python_speech_features` 0.6, `base.mfcc`, `base.fbank`,
  `sigproc.framesig`, `sigproc.powspec`.
- S. B. Davis and P. Mermelstein, "Comparison of parametric representations
  for monosyllabic word recognition in continuously spoken sentences", IEEE
  TASSP 28(4), 1980 (the original definition: 20 filters, magnitude spectrum,
  natural log, DCT without C0).

### What is and is not verified

The MATLAB preset is checked numerically against the `.mat` fixtures. The
other presets are checked structurally in `test/mfcc_variants.jl` (frame
counts, coefficient counts, energy terms, window and filterbank choices,
Float32 stability); no reference output of HTK, Kaldi, librosa, ETSI or
python_speech_features is available offline, so their numeric agreement is
documented from the sources above rather than tested. Two known differences
remain: Kaldi's dithering is disabled (it is random), and HTK and ETSI
operate on 16-bit integer samples, which the presets emulate by scaling
normalised audio by 32768 (`scale` keyword).
