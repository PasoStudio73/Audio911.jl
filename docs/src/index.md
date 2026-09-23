```@meta
CurrentModule = Audio911
```

```@raw html
<div align="center">
    <img src="../img/logo.png" alt="Audio911" width="600">
</div>
```

# Audio911.jl

**Audio911.jl** extracts audio features for machine learning in Julia:
spectrograms on linear, mel, bark and ERB scales, MFCC and GTCC cepstra in
every published variant, deltas, spectral and temporal descriptors, chroma,
onsets, tempo, harmonic/percussive separation and the utilities around
them, with numerical parity against MATLAB's Audio Toolbox where a fixture
exists and a design that lets you swap the algorithm at every stage.

## Highlights

- **Interchangeable front ends.** The short-time Fourier transform
  ([`Stft`](@ref)) and the wavelet scalogram ([`Cwt`](@ref)) implement one
  interface; every downstream stage (mel filterbanks, cepstra, descriptors)
  accepts either. See the [pipeline design](@ref design).
- **Every MFCC in the literature.** MATLAB, HTK, Kaldi, librosa, ETSI and
  python_speech_features recipes are one call each, and every axis they
  differ on is a keyword. See [MFCC variants](@ref mfcc_variants).
- **librosa and MATLAB coverage.** Feature by feature, with the status of
  each in the [coverage table](@ref coverage).
- **audioFlux's algorithms.** The constant-Q, S-, pseudo-wavelet and
  non-stationary Gabor transforms, reassignment and synchrosqueezing,
  discrete wavelets, Cohen-class and adaptive decompositions, spectral
  descriptors, pitch methods, NMF, HMMs, time stretch and pitch shift,
  checked against [audioFlux](https://github.com/libAudioFlux/audioFlux)
  itself. See the [audioFlux inventory](@ref audioflux).
- **Built-in audio loading.** WAV, FLAC, OGG and MP3 through libsndfile
  and mpg123; in-memory arrays through [`AudioFile`](@ref).
- **Plots recipes** for every stage, at no cost unless Plots is loaded.
- **Lean and type stable.** Lazy frames, a streamed pre-planned FFT,
  `Float32` end to end. See [performance](@ref performance).

## Installation

```julia
using Pkg
Pkg.add("Audio911")
```

## Quick start

```julia
using Audio911

audio = load("speech.wav"; sr=16000)                       # Float32, mono, resampled

stft  = Stft(audio; winsize=512, winstep=256, type=hamming)  # power spectrogram
mel   = MelSpec(stft; nbands=32, freqrange=(100, 8000))     # mel filterbank
mfcc  = Mfcc(mel; ncoeffs=13)                               # MATLAB-style MFCC
delta = Delta(mfcc)

get_data(mfcc)     # frames × 13
get_times(mfcc)    # frame centres in seconds

# same pipeline on a wavelet scalogram
mfcc_w = Mfcc(MelSpec(Cwt(audio; winsize=512, winstep=256); nbands=32); ncoeffs=13)

using Plots
plot(mel; freq_scale=:log10)
```

## Where to go next

- [Tutorial](@ref tutorial): the whole pipeline step by step.
- One page per stage: [Loading audio](@ref loading), [Frames](@ref frames),
  [STFT](@ref stft), [Wavelets](@ref cwt), [Filterbanks](@ref filterbanks),
  [Spectrograms](@ref spectrograms), [Cepstra](@ref cepstra),
  [Constant-Q and other transforms](@ref transforms),
  [Descriptors](@ref descriptors), [Features](@ref features),
  [Signal utilities](@ref signal), [Plotting](@ref plotting).
- [API reference](@ref api).

## About

Audio911.jl is developed by the [ACLAI Lab](https://aclai.unife.it/en/) at
the University of Ferrara. MIT license. The ports from
[audioFlux](https://github.com/libAudioFlux/audioFlux) (MIT licence,
Copyright (c) 2023 libAudioFlux) keep its attribution in their source
files.
