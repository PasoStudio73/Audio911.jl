```@meta
CurrentModule = Audio911
```

# [Spectrograms](@id spectrograms)

Filterbank stages multiply a front end by a filterbank designed on its
grid. They accept any `AbstractSpectrogram`.

| stage | filterbank | MATLAB feature |
|:------|:-----------|:---------------|
| [`LinSpec`](@ref) | none: frequency range, one-sided doubling, window normalisation | `linearSpectrum` |
| [`MelSpec`](@ref) | `auditory_fbank` with `htk` or `slaney` | `melSpectrum` |
| [`BarkSpec`](@ref) | `auditory_fbank` with `bark` | `barkSpectrum` |
| [`ErbSpec`](@ref) | `gammatone_fbank` | `erbSpectrum` |

```julia
mel = MelSpec(stft; win_norm=true, nbands=32, scale=htk, norm=bandwidth, domain=:linear, freqrange=(100, 8000))
mel = MelSpec(stft, auditory_fbank(stft; nbands=32); win_norm=true)
```

`win_norm=true` multiplies the filterbank by [`get_winnorm`](@ref) of the
front end (`1/sum(w)²` for power, `1/sum(w)` for magnitude, MATLAB's
`WindowNormalization`). [`get_data`](@ref) returns `frames × bands` (the
MATLAB orientation), [`get_spec`](@ref) the internal `bands × frames`.
[`get_freq`](@ref) gives band centres, [`get_fbank`](@ref) the filterbank,
[`get_nbands`](@ref), [`get_freqrange`](@ref) and [`get_parent`](@ref)
the rest.

[`DerivedSpec`](@ref) objects (the components of [`Hpss`](@ref), the output
of [`SpectralGate`](@ref) and [`pcen`](@ref)) are spectrograms on the grid
of their parent and go through the same stages.
