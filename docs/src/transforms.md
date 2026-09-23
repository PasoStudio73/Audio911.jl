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

## Reassigned spectrogram

[`Reassign`](@ref) moves every cell of an [`Stft`](@ref) to its local
centre of gravity (Auger & Flandrin 1995): the instantaneous frequency
`f − Im(S_dh / S_h) · sr / 2π` and the group delay `t + Re(S_th / S_h) / sr`,
where `S_dh` and `S_th` are the STFTs with the derivative of the window and
the time-weighted window. Cells weaker than `thresh` stay in place;
`mode=:freq` or `:time` reassigns one axis only and `order > 1` repeats the
frequency reassignment. With `accumulate=:complex` (audioFlux's default)
the complex values, re-referenced to the window centre, are added in their
new cell; `accumulate=:energy` adds their power instead and conserves the
total energy. [`get_reassigned`](@ref) returns the reassigned frequency and
time of every cell (librosa `reassigned_spectrogram`).

```julia
stft = Stft(audio; winsize=512, winstep=128, type=hanning)
r    = Reassign(stft)                    # sharper, same grid
f, t = get_reassigned(r)
plot(r; freq_scale=:log10)
```

The result keeps the STFT's frequency grid and frames, so it feeds every
downstream stage. It agrees with audioFlux to single precision
(`test/audioflux_files/reassign/`).

## Cohen-class distributions

[`Wvd`](@ref) is the pseudo Wigner-Ville distribution at the centre of
every frame: the lag product `z[n+m] conj(z[n-m])` of the analytic signal
([`hilbert`](@ref)), weighted by the frame window over the lag and Fourier
transformed. It has twice the frequency resolution of an STFT with the same
window, and interference terms between components. [`Cwd`](@ref), the
Choi-Williams distribution, first smooths the lag products over time with
the kernel `exp(-σ μ²/4m²)`, which attenuates those terms. Both return a
[`CohenDistribution`](@ref): `get_spec` is clipped at zero as the
interface requires, [`get_distribution`](@ref) is the signed distribution.
[`wvd`](@ref) is the whole-signal Wigner-Ville distribution for short
signals.

```julia
w = Wvd(Frames(audio; winsize=256, winstep=128, type=hanning))
c = Cwd(Frames(audio; winsize=256, winstep=128, type=hanning); sigma=0.5)
```

## Empirical mode decomposition and the Hilbert-Huang spectrum

[`emd`](@ref) sifts a signal into intrinsic mode functions (Huang et al.
1998): the mean of the cubic-spline envelopes through the maxima and the
minima is subtracted until the sifting converges, the fastest oscillation
first; the IMFs and the residual add up to the signal. [`Hht`](@ref) is
the Hilbert-Huang spectrum: the instantaneous power of every IMF at its
instantaneous frequency, pooled on the frames ([`get_imfs`](@ref) returns
the IMFs).

```julia
imfs, residual = emd(x)
h = Hht(frames; nbins=257)
```

## Empirical wavelet transform

[`ewt`](@ref) builds an adaptive wavelet filter bank (Gilles 2013): the
spectrum is segmented between its largest peaks and a Meyer-type tight
frame is placed on the segments, splitting the signal into `nbands`
components. [`Ewt`](@ref) pools the components on the frames.

```julia
components, bounds = ewt(x, 16000; nbands=5)
e = Ewt(frames; nbands=5)
```

audioFlux declares these five algorithms without implementing them; the
ports follow the cited papers and are tested structurally (marginals,
reconstruction, separation of known components).
