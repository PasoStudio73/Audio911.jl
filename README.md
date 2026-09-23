<div align="center">
    <img src="logo.png" alt="Audio911" width="600">
</div>

<h2 align="center">Audio Feature Extraction in Julia
<p align="center">
  <a href="https://github.com/aclai-lab/Audio911.jl/actions">
    <img src="https://github.com/aclai-lab/Audio911.jl/workflows/CI/badge.svg"
         alt="Build Status">
  </a>
  <a href="https://aclai-lab.github.io/Audio911.jl/dev/">
    <img src="https://img.shields.io/badge/docs-dev-blue.svg"
         alt="dev documentation">
  </a>
  <a href="https://aclai-lab.github.io/Audio911.jl/stable/">
    <img src="https://img.shields.io/badge/docs-stable-blue.svg"
         alt="stable documentation">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yelllow"
       alt="bibtex">
  </a>
  <a href="https://codecov.io/gh/aclai-lab/Audio911.jl">
    <img src="https://codecov.io/gh/aclai-lab/Audio911.jl/branch/main/graph/badge.svg"
       alt="CodeCov">
  </a>

</p>
</h2>

**Audio911.jl** extracts audio features for machine learning: spectrograms
on linear, mel, bark and ERB scales, MFCC and GTCC cepstra in every
published variant, deltas, spectral and temporal descriptors, chroma,
onsets and tempo, harmonic/percussive separation, and the utilities around
them. Where MATLAB's Audio Toolbox has a feature, Audio911 reproduces it
numerically (the test suite checks against `audioFeatureExtractor`
fixtures); where librosa has one, the coverage table in the docs says
whether and how it is covered. The algorithms of
[audioFlux](https://github.com/libAudioFlux/audioFlux) are ported and checked
against audioFlux itself; the inventory page lists every one of them.

## Features

### Interchangeable pipeline
Every stage is an immutable object built from the previous one, and every
time-frequency front end implements the same interface, so an STFT and a
wavelet scalogram feed the same mel filterbank, cepstrum and descriptors:

```
load ─▶ AudioFile ─▶ Frames ─┬─▶ Stft ──┐
                             ├─▶ Cwt ───┤
                             ├─▶ Cqt ───┼─▶ LinSpec ─────────────▶ Spectral*
                             ├─▶ Pwt ───┼─▶ MelSpec / BarkSpec ─▶ Mfcc ─▶ Delta
                             ├─▶ Nsgt ──┼─▶ ErbSpec ─────────────▶ Gtcc ─▶ Delta
                             └─▶ St/Fst ┴─▶ Chroma, Tonnetz, Contrast, Onset, Hpss, ...
```

### Time-frequency front ends
- **STFT** (`Stft`): pre-planned real FFT streamed over lazy frames, threaded,
  power or magnitude spectrum
- **Wavelet scalogram** (`Cwt`): Morlet, Morse, bump, Paul, DOG, Mexican hat,
  Hermitian and Ricker wavelets on a geometric or any filterbank grid, pooled
  on the same frame grid as the STFT; synchrosqueezing (`Wsst`, `Synsq`)
- **Constant-Q / variable-Q transform** (`Cqt`), **pseudo wavelet transform**
  (`Pwt`), **S-transform** and **fast S-transform** (`St`, `Fst`),
  **non-stationary Gabor transform** (`Nsgt`), ported from audioFlux and
  checked against it
- **Complex coefficients on request** (`get_complex`, `get_phase`), the
  inverse STFT (`istft`) and the **reassigned spectrogram** (`Reassign`)

### Spectrograms and filterbanks
- `LinSpec`, `MelSpec` (HTK or Slaney mel), `BarkSpec`, `ErbSpec` (gammatone)
- audioFlux's `linspace`, `erb`, `octave` and `logspace` scales and window
  filter styles (`hanning`, `gauss`, `kaiser`, `bohman`, `point`, ...) on any
  filterbank stage
- filterbank design on any frequency grid: `auditory_fbank`, `gammatone_fbank`,
  `chroma_fbank`, plus the integer-bin banks of ETSI and python_speech_features

### Cepstra, in every variant
- `Mfcc`, `Gtcc`, `Delta`, with every axis the literature differs on as a
  keyword (rectification, floor, DCT scaling, liftering, C0, energy)
- presets: `mfcc_matlab`, `mfcc_htk`, `mfcc_kaldi`, `mfcc_librosa`,
  `mfcc_etsi`, `mfcc_psf`

### Descriptors and features
- eleven MATLAB spectral descriptors plus `SpectralBandwidth`, `Rms`, `Energy`,
  `Zcr`, `Pitch` (NCF, YIN, cepstral), `HarmonicRatio`
- `Chroma`, `Tonnetz`, `SpectralContrast`, `PolyFeatures`
- `OnsetStrength`, `onset_detect`, `Tempogram`, `tempo`, `beat_track`
- `Hpss` harmonic/percussive separation, `pcen`, `noisegate`, `SpectralGate`
- unit conversions, dB scaling, weighting curves, synthesis, silence trimming, LPC

### Extras
- **Built-in audio loading**: WAV, FLAC, OGG and MP3 through libsndfile and
  mpg123; in-memory arrays through `AudioFile(x, sr)`; resampling, mono, normalisation
- **Plots recipes** for every stage (`using Plots; plot(mel)`), no Plots dependency
- **Float32 end to end** and allocation-regression tests

## Installation

```julia
using Pkg
Pkg.add("Audio911")
```

## Quick Start

```julia
using Audio911

# Load an audio file (resampled to 16 kHz, mono, Float32)
audio = load("speech.wav"; sr=16000)

# Frames and STFT
stft = Stft(audio; winsize=512, winstep=256, type=hamming, periodic=true, spectrum=power)

# Mel spectrogram (MATLAB-compatible defaults)
mel = MelSpec(stft; win_norm=true, nbands=32, norm=bandwidth, domain=:linear, scale=htk)

# MFCC and deltas
mfcc  = Mfcc(mel; ncoeffs=13, rect=mlog)
delta = Delta(mfcc)

get_data(mfcc)      # frames × 13 matrix
get_times(mfcc)     # frame centres in seconds

# Same pipeline on a wavelet scalogram
mfcc_w = Mfcc(MelSpec(Cwt(audio; winsize=512, winstep=256); nbands=32); ncoeffs=13)

# A published recipe in one call
k = mfcc_kaldi(audio)

using Plots
plot(mel; freq_scale=:log10)
```

## Learn More

The [documentation](https://aclai-lab.github.io/Audio911.jl/dev/) has a
step-by-step tutorial, one page per stage, the pipeline design, the MFCC
variants with their references, the librosa/MATLAB coverage table,
before/after performance numbers and the full API reference.

Development: `test/run.sh test` runs the suite, `test/run.sh docs` builds
the site, `test/run.sh bench` prints the benchmark.

## About

Audio911.jl is developed by the [ACLAI Lab](https://aclai.unife.it/en/) @ University of Ferrara.

## License

MIT License
