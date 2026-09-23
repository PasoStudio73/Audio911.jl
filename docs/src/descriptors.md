```@meta
CurrentModule = Audio911
```

# [Descriptors](@id descriptors)

One value per frame. Spectral descriptors take any spectrogram (a
[`LinSpec`](@ref) with a frequency range, a [`MelSpec`](@ref), a
[`Cwt`](@ref), ...); time-domain descriptors take [`Frames`](@ref). All are
`AbstractSpectral`, keep their parent and share its time axis.

| descriptor | reference |
|:-----------|:----------|
| [`SpectralCentroid`](@ref), [`SpectralSpread`](@ref), [`SpectralSkewness`](@ref), [`SpectralKurtosis`](@ref) | MATLAB spectral moments |
| [`SpectralCrest`](@ref), [`SpectralDecrease`](@ref), [`SpectralEntropy`](@ref), [`SpectralFlatness`](@ref), [`SpectralSlope`](@ref) | MATLAB |
| [`SpectralFlux`](@ref) (`p`), [`SpectralRolloff`](@ref) (`threshold`) | MATLAB |
| [`SpectralBandwidth`](@ref) (`p`) | librosa `spectral_bandwidth` |
| [`Rms`](@ref) | librosa `rms` (frames or spectrogram) |
| [`Energy`](@ref) | MATLAB `shortTimeEnergy` |
| [`Zcr`](@ref) | librosa `zero_crossing_rate`, MATLAB `zerocrossrate` |
| [`Pitch`](@ref) with [`pitch_ncf`](@ref), [`pitch_yin`](@ref), [`pitch_cep`](@ref) | MATLAB `pitch` (NCF, CEP), librosa `yin` |
| [`HarmonicRatio`](@ref) | MATLAB `harmonicRatio` |
| [`OnsetStrength`](@ref) | librosa `onset_strength` |

```julia
lin = LinSpec(stft; freqrange=(100, 4000), win_norm=true)
c   = SpectralCentroid(lin)
get_data(c), get_times(c)
f0  = Pitch(frames; method=pitch_yin, range=(60, 400))
```

The eleven MATLAB descriptors are checked against `audioFeatureExtractor`
fixtures on a linear spectrum.
