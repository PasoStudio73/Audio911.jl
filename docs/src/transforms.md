```@meta
CurrentModule = Audio911
```

# [Constant-Q and other transforms](@id transforms)

The front ends on this page were ported from
[audioFlux](https://github.com/libAudioFlux/audioFlux) (MIT licence); the
[inventory](@ref audioflux) lists what each one matches. All of them
implement the [front-end interface](@ref design), so a mel filterbank, a
cepstrum, the descriptors or a chromagram accept them exactly like an
[`Stft`](@ref).

## Constant-Q and variable-Q transform

[`Cqt`](@ref) is the spectral-kernel constant-Q transform (Brown &
Puckette 1992): `bins_per_octave` bins per octave from `fmin`, every bin a
windowed complex exponential whose length `Q · sr / f` keeps the ratio of
centre frequency to bandwidth constant. The kernels are transformed once,
thresholded to a sparse spectral kernel and applied to the FFT of an
analysis window centred on every frame of the `Frames` object; the frame
length of `Frames` only fixes the time grid. With `gamma > 0` the kernel
length becomes `Q · sr / (f + γ / (2^(1/b) − 1))`, the variable-Q transform
of Schörkhuber et al. (2014).

```julia
frames = Frames(audio; winsize=512, winstep=256, center=true)
cqt    = Cqt(frames; fmin=note_to_hz("C1"), bins_per_octave=12, nbins=84)
vqt    = Cqt(frames; gamma=20)
chroma = Chroma(cqt)                 # folds the bins of every octave
cqcc   = Mfcc(cqt; ncoeffs=13)       # constant-Q cepstral coefficients
plot(cqt)                            # log-frequency heatmap
```

`norm` selects the kernel normalisation (`area`, `none_norm`, `bandwidth`),
`scale` divides every bin by the square root of its kernel length,
`keep_complex=true` keeps the complex coefficients for [`get_complex`](@ref)
(they are recomputed on request otherwise). [`get_nfft`](@ref) returns the
FFT window and [`get_bandwidth`](@ref) the kernel lengths.

audioFlux evaluates the lower octaves on a 2:1 decimated copy of the
signal and reuses the top octave's kernel lengths for them (so its
variable-Q transform is variable-Q in the top octave only); Audio911
evaluates every bin on the full-rate signal with its own length. The
fixtures in `test/audioflux_files/cqt/` therefore agree to single
precision on the top octave and within a few percent below it.

## Frequency scales and filter styles

The filterbank stages gained audioFlux's scales and band shapes. `scale`
can now be [`linspace`](@ref) (linear), [`erb`](@ref) (Glasberg–Moore
ERB-rate, triangular filters), [`octave`](@ref) (`bins_per_octave` bands
per octave anchored on 440 Hz) or [`logspace`](@ref) (geometric) next to
`htk`, `slaney` and `bark`; `style` shapes every band as a triangle
([`triangular`](@ref), the default), a bin-based triangle ([`etsi`](@ref)),
a single bin ([`point`](@ref)), a rectangle (`rect`) or a `hanning`, `hamming`, `blackman`,
[`bohman`](@ref), [`kaiser`](@ref) or [`gauss`](@ref) window between the
neighbouring centres.

```julia
mel = MelSpec(stft; nbands=40, scale=erb, style=gauss)
oct = MelSpec(stft; nbands=84, scale=octave, bins_per_octave=12, freqrange=(33, 8000))
```

For `linspace`, `octave` and `logspace` the `freqrange` bounds are the
first and last band centre (audioFlux's convention); for the other scales
they are the outer edges.

## Pseudo wavelet transform

[`Pwt`](@ref) multiplies the FFT of the whole signal by a bank of bandpass
filters designed with [`auditory_fbank`](@ref) on the signal's own FFT grid
(any scale, any style, any normalisation), inverse-transforms every band
and pools the power over the frames like [`Cwt`](@ref). The
function form [`pwt`](@ref) returns the complex band series
(`bands × samples`).

```julia
p   = Pwt(frames; nbands=84, scale=octave, style=hanning, norm=bandwidth)
Y, f = pwt(get_signal(frames), 16000; nbands=40, scale=htk)
```

## S-transform and fast S-transform

[`St`](@ref) is the Stockwell transform: for every FFT bin `k` the spectrum
is shifted by `k`, multiplied by a Gaussian voice whose width scales with
`k` (`factor` and `norm` shape it) and inverse-transformed, one row per bin
of `freqrange`. [`st`](@ref) is the whole-signal form. The transform costs
one inverse FFT per bin, so restrict `freqrange` on long signals.

[`Fst`](@ref) / [`fst`](@ref) is the fast S-transform of Brown, Lauzon and
Frayne (2010): the spectrum of a `2^m`-sample signal is cut into dyadic
blocks, every block is inverse-transformed at its own length and its
samples are held over the time axis, which samples the S-transform
non-redundantly. The front end zero-pads the signal to a power of two.

```julia
s = St(frames; freqrange=(0, 500))
f = Fst(frames; freqrange=(0, 4000))
```

## Non-stationary Gabor transform

[`Nsgt`](@ref) / [`nsgt`](@ref) cut the whole-signal spectrum into one
window per band (edges from any scale, window shape from `style`, the
efficient symmetric bank or the standard periodic one), demodulate every
slice to baseband and inverse-transform it at its own length, so each band
keeps its own time resolution. [`get_cells`](@ref) returns those series,
[`get_lengths`](@ref) their lengths, and [`nsgt_matrix`](@ref) resamples
them to a common length the way audioFlux does. The front end expands
every cell to the signal length by the same hold rule and pools it over
the frames.

```julia
n = Nsgt(frames; nbands=84, scale=octave, style=hanning, norm=bandwidth)
cells, freq, lengths = nsgt(get_signal(frames), 16000)
M = nsgt_matrix(cells, lengths, length(get_signal(frames)) / 16000, maximum(lengths))
```
