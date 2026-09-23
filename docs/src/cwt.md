```@meta
CurrentModule = Audio911
```

# [Wavelets](@id cwt)

[`Cwt`](@ref) is the wavelet alternative to the STFT. An analytic mother
wavelet ([`morlet`](@ref), [`morse`](@ref) or [`bump`](@ref), or any
function of the angular frequency) is applied scale by scale in the
frequency domain, with one inverse FFT per scale and L1 normalisation. The
scales are spaced geometrically, `voices` per octave between `freqrange`.
The squared modulus is then pooled over the frames of the `Frames` object
(weighted by its window), so the scalogram has exactly one column per frame
and can replace an STFT anywhere:

```julia
frames = Frames(audio; winsize=512, winstep=256, type=rect)
cwt    = Cwt(frames; wavelet=morse, voices=16, freqrange=(50, 8000))
mel    = MelSpec(cwt; nbands=26)
mfcc   = Mfcc(mel; ncoeffs=13)
chroma = Chroma(cwt)
```

Eight analytic wavelets are available: [`morlet`](@ref), [`morse`](@ref),
[`bump`](@ref), [`paul`](@ref), [`dog`](@ref), [`mexican`](@ref),
[`hermit`](@ref) and [`ricker`](@ref) (the last five ported from
audioFlux), or any function of the angular frequency. A scale is mapped to
a frequency through the wavelet's centre angular frequency (`centre`,
audioFlux's value for the built-in wavelets, the spectral peak otherwise).

The default grid is geometric, `voices` per octave between `freqrange`.
With `scale` (and `nbands`, `bins_per_octave`) the bands sit on any
filterbank scale instead, audioFlux's grids: `octave` from C1, `htk` mel,
`bark`, `erb`, `linspace`, `logspace`:

```julia
cwt_p = Cwt(frames; wavelet=paul, scale=octave, nbands=84, freqrange=(33, 8000))
W, f  = cwt(get_signal(frames), 16000; wavelet=morse, scale=htk, nbands=64, freqrange=(0, 8000))
```

[`cwt`](@ref) is the whole-signal complex transform (`bands × samples`,
audioFlux's `CWT`, symmetric padding with `pad=true`) and
[`get_complex`](@ref) returns the coefficients of a `Cwt` at its frame
centres. [`get_freq`](@ref) returns the centre frequency of every scale
(ascending) and [`get_scales`](@ref) the scales in samples. A scalogram has no analysis
window, so `win_norm` on the stages fed by it is a no-op
([`get_winnorm`](@ref) returns 1). A `bump` wavelet with many voices gives a
constant-Q-like representation; librosa's CQT kernels are not reproduced.

Edge effects are handled by reflecting the signal by four times the
wavelet's time support at the lowest frequency, and the transform is
threaded over scales.

## Synchrosqueezing

A synchrosqueezed transform (Daubechies, Lu & Wu 2011) moves every wavelet
coefficient, at its own time, to the band nearest to its instantaneous
frequency, which sharpens the ridges of a scalogram without changing its
grid. [`Wsst`](@ref) estimates the instantaneous frequency from the
derivative wavelet (`|Im(W_∂/W)| · sr/2π`, audioFlux `wsst`),
[`Synsq`](@ref) from the phase difference of consecutive coefficients
(audioFlux `synsq`). Both return a [`Synchrosqueezed`](@ref) spectrogram on
the grid and frames of the `Cwt`, so the downstream stages accept it.

```julia
c = Cwt(frames; wavelet=morse, scale=octave, nbands=84, freqrange=(33, 8000))
w = Wsst(c)                         # or Wsst(frames; nbands=84)
s = Synsq(c; order=2)
S, W, f = wsst(get_signal(frames), 16000; nbands=84)   # whole-signal, complex
Q = synsq(W, f, 16000)
```

The front ends add the power of the moved coefficients
(`accumulate=:energy`), so their memory is the size of the output;
`accumulate=:complex` adds the complex coefficients first, as audioFlux
does, at the cost of a `bands × samples` buffer. The whole-signal
[`wsst`](@ref) and [`synsq`](@ref) always accumulate complex values.
Coefficients map to their nearest band (in log frequency on geometric
grids); audioFlux's octave and linear grids use a step of `(n-1)/n` bins
instead, and its `synsq` unwraps the phase in single precision, see the
[inventory](@ref audioflux).
