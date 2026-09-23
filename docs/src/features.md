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
- [`onset_detect`](@ref), [`peak_pick`](@ref): onset frames.
- [`Tempogram`](@ref), [`tempo`](@ref): local auto-correlation and global tempo.
- [`beat_track`](@ref): dynamic-programming beat tracking.

```julia
env = OnsetStrength(MelSpec(Stft(audio; winsize=2048, winstep=512, center=true); nbands=64))
onsets = get_times(env)[onset_detect(env)]
bpm, beats = beat_track(env)
```

## Decomposition and gates

- [`Hpss`](@ref): harmonic/percussive components as [`DerivedSpec`](@ref)s
  ([`get_harmonic`](@ref), [`get_percussive`](@ref), [`get_masks`](@ref)).
- [`pcen`](@ref): per-channel energy normalisation.
- [`noisegate`](@ref): MATLAB-style time-domain gate with attack, release and hold.
- [`SpectralGate`](@ref): the same idea on a frequency range of a spectrogram.

```julia
h   = Hpss(stft; kernel=(31, 31), power=2)
mel = MelSpec(get_harmonic(h); nbands=40)
g   = SpectralGate(stft; threshold=-50, freqrange=(40, 120))   # gate the hum band only
```
