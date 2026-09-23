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
| [`SpectralFlux`](@ref) (`p`; audioFlux's `step`, `positive`, `root`, `mean`), [`SpectralRolloff`](@ref) (`threshold`) | MATLAB, audioFlux `flux` |
| [`SpectralBandwidth`](@ref) (`p`, `normalize`) | librosa `spectral_bandwidth`, audioFlux `band_width` |
| [`SpectralEnergy`](@ref), [`SpectralRms`](@ref), [`SpectralHfc`](@ref), [`SpectralMax`](@ref), [`SpectralPeak`](@ref), [`SpectralMean`](@ref), [`SpectralVar`](@ref) | audioFlux `energy`, `rms`, `hfc`, `max`, `mean`, `var` |
| [`SpectralSd`](@ref), [`SpectralSf`](@ref), [`SpectralMkl`](@ref), [`SpectralBroadband`](@ref), [`SpectralNovelty`](@ref) | audioFlux onset novelties (spectral difference, squared difference, modified Kullback-Leibler, broadband, `novelty`) |
| [`SpectralPd`](@ref), [`SpectralWpd`](@ref), [`SpectralNwpd`](@ref), [`SpectralCd`](@ref), [`SpectralRcd`](@ref) | audioFlux phase and complex-domain deviations (need a front end with [`get_complex`](@ref)) |
| [`SpectralEef`](@ref), [`SpectralEer`](@ref) | audioFlux energy-entropy feature and ratio |
| [`Rms`](@ref) | librosa `rms` (frames or spectrogram) |
| [`Energy`](@ref) | MATLAB `shortTimeEnergy` |
| [`Zcr`](@ref) | librosa `zero_crossing_rate`, MATLAB `zerocrossrate` |
| [`Pitch`](@ref) with [`pitch_ncf`](@ref), [`pitch_yin`](@ref), [`pitch_cep`](@ref), [`pitch_pef`](@ref), [`pitch_hps`](@ref), [`pitch_lhs`](@ref), [`pitch_stft`](@ref) | MATLAB `pitch` (NCF, CEP, PEF, LHS), librosa `yin`, audioFlux `PitchHPS`, `PitchSTFT` |
| [`HarmonicRatio`](@ref) | MATLAB `harmonicRatio` |
| [`OnsetStrength`](@ref) | librosa `onset_strength` |
| [`Novelty`](@ref) | audioFlux `Onset` envelope: any novelty above, min–max normalised |

```julia
lin = LinSpec(stft; freqrange=(100, 4000), win_norm=true)
c   = SpectralCentroid(lin)
get_data(c), get_times(c)
f0  = Pitch(frames; method=pitch_yin, range=(60, 400))
```

The eleven MATLAB descriptors are checked against `audioFeatureExtractor`
fixtures on a linear spectrum, and every audioFlux descriptor against
audioFlux's `Spectral` on a magnitude STFT (`test/spectral_af.jl`). Where
the two libraries define a descriptor differently (the flux root, the
entropy normalisation, the bandwidth weighting), the MATLAB definition is
the default and audioFlux's is a keyword.

```julia
stft = Stft(audio; winsize=512, winstep=256, keep_complex=true)
SpectralFlux(stft; root=false, positive=true)      # audioFlux's flux
SpectralNovelty(stft; method=:kl, data=:number)
SpectralCd(stft)                                    # uses the phase
```
