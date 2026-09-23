```@meta
CurrentModule = Audio911
```

# [STFT](@id stft)

[`Stft`](@ref) is the default time-frequency front end: every frame is
windowed, zero-padded to `nfft`, transformed with a pre-planned real FFT
and reduced to a one-sided [`power`](@ref) (`|X|²`) or
[`magnitude`](@ref) (`|X|`) spectrum. The work is streamed frame by frame
with one buffer per thread, so memory is the output matrix plus a few
`nfft`-length buffers.

```julia
stft = Stft(frames; nfft=1024, spectrum=power)
stft = Stft(audio; winsize=512, winstep=256, type=hamming, nfft=512)
```

The frequency grid is `0:sr/nfft:sr/2` ([`get_freq`](@ref)); the data is
`bins × frames` ([`get_spec`](@ref), also [`get_data`](@ref) for this
type). [`get_nfft`](@ref), [`get_spectrum`](@ref), [`get_window`](@ref),
[`get_winsize`](@ref), [`get_step`](@ref), [`get_overlap`](@ref),
[`get_times`](@ref) and [`get_frames`](@ref) expose the parameters.
`scale` multiplies the spectrum by a constant (python_speech_features uses
`1/nfft`).

The complex STFT is not kept, which is why inverse transforms are not
offered; see the [coverage table](@ref coverage).
