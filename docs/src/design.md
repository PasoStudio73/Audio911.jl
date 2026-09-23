```@meta
CurrentModule = Audio911
```

# [Pipeline design](@id design)

Audio911 is a chain of **immutable stage objects**. Every stage is built from
the previous one and carries, next to its data, an `info` record with the
parameters that produced it, so a downstream stage never asks the user for a
value that an upstream stage already knows.

```
load ─▶ AudioFile ─▶ Frames ─┬─▶ Stft ──┐
                             ├─▶ Cwt ───┤
                             ├─▶ Cqt ───┼─▶ LinSpec ─────────────▶ Spectral*
                             ├─▶ Pwt ───┤   MelSpec / BarkSpec ─▶ Mfcc ─▶ Delta
                             ├─▶ Nsgt ──┤   ErbSpec ─────────────▶ Gtcc ─▶ Delta
                             └─▶ St/Fst ┘   Chroma, Contrast, Tonnetz, Onset, ...
```

The point of the design is the vertical bar in the middle: **any
time-frequency front end can feed any downstream stage**. `MelSpec`, `LinSpec`,
`Mfcc`, the spectral descriptors and the newer features are written against an
abstract interface, not against `Stft`. A scalogram computed with wavelets
(`Cwt`) feeds a mel filterbank exactly like an STFT does, and a new front end
(a constant-Q transform, a gammatone filterbank output, a reassigned
spectrogram) only has to implement the interface below.

## The front-end interface

A front end is any subtype of `AbstractSpectrogram`. Downstream stages rely on
exactly these accessors:

| accessor            | returns                                   | required |
|:--------------------|:------------------------------------------|:---------|
| `get_spec(s)`       | `AbstractMatrix{T}`, **bins × frames**, real, non-negative | yes |
| `get_freq(s)`       | `AbstractVector`, ascending Hz, `length == size(get_spec(s), 1)` | yes |
| `get_sr(s)`         | sample rate in Hz (`Int`)                 | yes |
| `get_spectrum(s)`   | `power` or `magnitude`                    | yes |
| `get_window(s)`     | the analysis window as a vector, or `nothing` | default `nothing` |
| `get_winnorm(s)`    | scalar window-normalisation factor        | default derived from `get_window` |
| `get_step(s)`       | hop between frames, in samples            | yes (time axis) |
| `get_winsize(s)`    | frame length in samples                   | yes (time axis) |
| `get_times(s)`      | frame centre times in seconds             | default derived |
| `Base.eltype(s)`    | element type `T`                          | yes |

- **Orientation.** Internally every matrix is `bins × frames` (column major,
  one frame per column) and `get_spec` always returns that orientation.
  `get_data` is the *public* view and keeps its historical orientation:
  `Stft` and `FBank` return the raw `bins × frames`, while `LinSpec`, `MelSpec`,
  `BarkSpec`, `ErbSpec`, `Mfcc`, `Gtcc` return the adjoint `frames × bands` so
  that results line up with MATLAB's `extract` output. New code that consumes
  a spectrogram must use `get_spec`.
- **Frequency grid.** `get_freq` may be non-uniform (a scalogram uses a
  geometric grid). Filterbank design (`auditory_fbank`, `gammatone_fbank`,
  `chroma_fbank`) takes the grid itself (`sfreq`) and never assumes an FFT
  spacing. `Stft` additionally exposes `get_nfft` because its grid is an FFT
  grid; nothing downstream requires it.
- **Spectrum kind.** `get_spectrum` tells filterbank stages whether the bins
  hold power (`|X|²`) or magnitude (`|X|`). It is used to pick the window
  normalisation (`winpower` versus `winmagnitude`) and by features that need a
  magnitude (they take the square root of a power spectrum instead of asking
  for a second transform).
- **Window hooks.** `get_window` returns the time-domain window when one
  exists. `get_winnorm(s)` is the scalar by which a filterbank is multiplied
  when `win_norm=true`; the default implementation returns
  `winpower(1, w)` or `winmagnitude(1, w)` for a windowed front end and `1`
  when `get_window(s) === nothing`, so `win_norm` is a safe no-op for a
  front end without a window.
- **Element type.** `load(...; format=Float32)` fixes `T` and every stage is
  parameterised on it. Windows, filterbanks, DCT matrices and delta filters are
  built in `T`; nothing in the pipeline promotes `Float32` to `Float64`. A test
  (`test/type_stability.jl`) walks the whole chain in both types.

## Stages that consume the interface

| stage                  | reads                                          |
|:-----------------------|:-----------------------------------------------|
| `LinSpec(s)`           | `get_spec`, `get_freq`, `get_sr`, `get_spectrum`, `get_winnorm` |
| `MelSpec(s, fbank)`    | `get_spec`, `get_spectrum`, `get_winnorm`; the filterbank must have `size(fb, 2) == length(get_freq(s))` |
| `MelSpec(s; kw...)`    | designs the filterbank on `get_freq(s)` then calls the above |
| `BarkSpec`, `ErbSpec`  | same as `MelSpec`, with the bark scale or a gammatone design pinned |
| `Mfcc(s; kw...)`       | any `AbstractSpectrogram` (it is a cepstrum of whatever bands it is given) |
| `Gtcc(s)`              | `ErbSpec` only (by definition) |
| `Spectral*(s)`         | `get_spec`, `get_freq`, `get_sr` |
| `Chroma`, `SpectralContrast`, `OnsetStrength`, ... | `get_spec`, `get_freq`, `get_sr`, `get_spectrum` |

`Delta` consumes any `AbstractAudioSpectrum` that has a `frames × bands`
`get_data` (cepstra, deltas, spectrograms).

## Front ends shipped

- **`Stft`** – the short-time Fourier transform. Frames are windowed and
  transformed with a pre-planned real FFT, one frame at a time, straight into
  the output matrix. Its grid is uniform, `sr/nfft` apart.
- **`Cwt`** – a continuous wavelet transform scalogram. The analytic wavelet
  (`morlet`, `morse` or `bump`) is evaluated in the frequency domain and
  applied scale by scale with one inverse FFT per scale; the squared modulus is
  then pooled over the same `winsize`/`winstep` grid as `Stft`, so the
  scalogram has the same number of frames as the equivalent STFT and can be
  swapped for it without touching the rest of the pipeline. Its grid is
  geometric (`voices` per octave between `freqrange`).

```julia
audio = load("speech.wav"; sr=16000)

# STFT route
mel_a = MelSpec(Stft(audio; winsize=512, winstep=256); nbands=26)
# wavelet route: same downstream code
mel_b = MelSpec(Cwt(audio; winsize=512, winstep=256, voices=12); nbands=26)

mfcc_a, mfcc_b = Mfcc(mel_a), Mfcc(mel_b)
```

- **`Cqt`** – the constant-Q (and variable-Q) transform. Sparse spectral
  kernels are applied to an FFT window centred on every frame; the top
  octave uses the shortest power-of-two window that holds its kernels and
  every octave below doubles it. The `Frames` object fixes only the time
  grid. Its grid is geometric (`bins_per_octave` per octave from `fmin`).
- **`Pwt`, `St`, `Fst`, `Nsgt`** – whole-signal transforms (pseudo wavelet,
  S-transform, fast S-transform, non-stationary Gabor). The whole signal is
  transformed once and every band is pooled over the frames exactly like
  `Cwt`, so each has one column per frame. `Nsgt` also keeps its cells, the
  band series at their own time resolution.

## [Complex coefficients on request](@id design_complex)

A front end stores only the real `power` or `magnitude` matrix, which is
what every downstream stage reads and what bounds the memory of a pipeline.
Phase-aware stages (reassignment, synchrosqueezing, phase-deviation
onsets, the phase vocoder, inverse transforms) need the complex
coefficients, so the interface has a second, optional accessor:

| accessor           | returns                                          |
|:-------------------|:-------------------------------------------------|
| `get_complex(s)`   | `Matrix{Complex{T}}`, **bins × frames**, the coefficients `get_spec` was derived from |
| `get_phase(s)`     | `angle.(get_complex(s))`                          |

The contract keeps the existing paths unchanged:

- **Nothing is kept by default.** A front end built as before holds the
  same fields and the same memory; `get_complex` recomputes the
  coefficients from its `Frames` (which keep the signal) when asked.
- **`keep_complex=true`** at construction stores the complex matrix next
  to the real one, for callers that will ask for it repeatedly. The real
  spectrogram is then derived from it, so the two are always consistent.
- **`Stft`** recomputes the complex STFT frame by frame into one matrix,
  or keeps it with `keep_complex=true`; [`istft`](@ref) inverts it.
- **Derived front ends** such as [`Reassign`](@ref) read `get_complex` of
  their parent and store only their own result.
- **Pooled front ends** (`Pwt`, `St`, `Fst`, `Nsgt`) have one
  complex series per band at the full signal rate. Their `get_complex`
  returns those series sampled at the frame centres, which is the value
  the phase-aware stages need on the frame grid.

The real matrices, their orientation and the MATLAB parity fixtures are not
affected: `get_spec` never goes through the complex path.

## Frames are lazy

`Frames` does not copy the signal into a `winsize × nframes` matrix. It stores
the signal, the frame starts and the window, and every consumer streams
through the frames with one preallocated buffer. `get_data(frames)` still
returns the materialised matrix for users who want it. Optional per-frame
pre-emphasis, DC removal and centre padding (librosa style) live here because
they are frame-level operations in every reference implementation.

## Options are functions

Configuration values are the functions themselves and are compared by
identity: window `hamming`/`hanning`/`povey`/..., spectrum
`power`/`magnitude`, scale `htk`/`slaney`/`bark`, filterbank normalisation
`area`/`bandwidth`/`none_norm`, rectification `mlog`/`nlog`/`cubic_root`/`db`.
The only Symbol-valued option is `domain=:linear|:warped` on filterbank design.

## What stays fixed

Numerical parity with MATLAB's `audioFeatureExtractor` is the test oracle:
every fixture under `test/matlab_files/` is a contract and the refactor is
built around it. Public constructor names and the `get_*` accessor family are
kept; new behaviour is added with new keywords or new accessors.
