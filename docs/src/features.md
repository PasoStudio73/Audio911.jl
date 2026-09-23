```@meta
CurrentModule = Audio911
```

# [Features](@id features)

Matrix-valued features and decompositions, all written against the
front-end interface.

## Tonal

- [`Chroma`](@ref) / [`chroma_fbank`](@ref): pitch-class energy (librosa
  `chroma_stft`), any front end, per-frame normalisation.
- [`Tonnetz`](@ref): six tonal centroids from a chromagram.
- [`SpectralContrast`](@ref): octave-band peak/valley contrast.
- [`PolyFeatures`](@ref): polynomial fit of every frame against frequency.

## Rhythm

- [`OnsetStrength`](@ref): spectral-flux envelope (use a `MelSpec` to match librosa).
- [`Novelty`](@ref): audioFlux's onset envelope from any spectral novelty
  (flux, HFC, spectral difference, phase or complex-domain deviation, ...).
- [`onset_detect`](@ref), [`peak_pick`](@ref): onset frames from either envelope.
- [`Tempogram`](@ref), [`tempo`](@ref): local auto-correlation and global tempo.
- [`beat_track`](@ref): dynamic-programming beat tracking.

```julia
env = OnsetStrength(MelSpec(Stft(audio; winsize=2048, winstep=512, center=true); nbands=64))
onsets = get_times(env)[onset_detect(env)]
nov = Novelty(Stft(audio; winsize=512, winstep=256, keep_complex=true); method=SpectralCd)
onsets_cd = get_times(nov)[onset_detect(nov)]
bpm, beats = beat_track(env)
```

## Decomposition and gates

- [`Hpss`](@ref): harmonic/percussive components as [`DerivedSpec`](@ref)s
  ([`get_harmonic`](@ref), [`get_percussive`](@ref), [`get_masks`](@ref)); on an
  `Stft`, the separated signals ([`get_harmonic_signal`](@ref),
  [`get_percussive_signal`](@ref)). `edge=:zero` pads the median filters
  with zeros as audioFlux does.
- [`nmf`](@ref): non-negative matrix factorisation of a spectrogram into
  spectral templates and activations (KL, Itakura–Saito or Euclidean).
- [`pcen`](@ref): per-channel energy normalisation.
- [`noisegate`](@ref): MATLAB-style time-domain gate with attack, release and hold.
- [`SpectralGate`](@ref): the same idea on a frequency range of a spectrogram.

```julia
h   = Hpss(stft; kernel=(31, 31), power=2)
mel = MelSpec(get_harmonic(h); nbands=40)
yh  = get_harmonic_signal(h)                               # back to the time domain
W, H = nmf(Stft(audio; spectrum=magnitude), 8)            # templates × activations
g   = SpectralGate(stft; threshold=-50, freqrange=(40, 120))   # gate the hum band only
```

## audioFlux features

- [`Deconv`](@ref) splits every frame of a spectrogram into a timbre
  (formant) part and a pitch (harmonic fine structure) part by
  deconvolution: the timbre is `real(ifft(|F|))` and the pitch
  `real(ifft(F/|F|))` of the Fourier transform `F` of the frame's spectrum
  ([`get_timbre`](@ref), [`get_pitch`](@ref)).
- [`Cepstrogram`](@ref) is the real cepstrum of every frame, with the
  log-power envelope rebuilt from the low quefrencies and the details from
  the high ones ([`get_envelope`](@ref), [`get_details`](@ref),
  [`get_quefrency`](@ref)).
- [`Ezr`](@ref) is the energy to zero-crossing ratio of every windowed
  frame; [`Zcr`](@ref) gained `windowed` and `strict` for audioFlux's
  temporal zero-crossing rate.
- [`HarmonicRatio`](@ref) with `method=:audioflux` is audioFlux's harmonic
  ratio (zero-padded auto-correlation from its first zero crossing,
  quadratic interpolation).

```julia
d  = Deconv(Stft(audio; winsize=512, winstep=256, spectrum=magnitude))
cg = Cepstrogram(Frames(audio; winsize=512, winstep=256, type=rect); ncep=8)
hr = HarmonicRatio(Frames(audio; winsize=4096, winstep=1024, type=hamming); method=:audioflux, fmin=32.703)
plot(d); plot(cg)
```

All four agree with audioFlux (`test/af_features.jl`).
