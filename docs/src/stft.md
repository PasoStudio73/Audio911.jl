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

## Complex STFT and inverse

The complex STFT is not stored unless asked for:
[`get_complex`](@ref) recomputes it from the frames (or returns the matrix
kept with `keep_complex=true`), and [`get_phase`](@ref) gives its phase.
The real spectrogram, and therefore every MATLAB parity result, is the
same either way (see [complex coefficients on request](@ref design_complex)).

[`istft`](@ref) inverts it by weighted overlap-add (`method=:wola`, the
default of audioFlux, librosa and MATLAB) or plain overlap-add
(`method=:ola`). With centred frames and a window that satisfies the
overlap condition (Hann at 50 % or 75 % overlap) the reconstruction is
exact:

```julia
stft = Stft(audio; winsize=1024, winstep=256, center=true, keep_complex=true)
y    = istft(stft)                                  # ≈ get_data(audio)
y2   = istft(get_complex(stft), 1024, 256; window=hanning, offset=-512)
```

The phase-aware stages ([`Reassign`](@ref), the phase-deviation onset
descriptors, the phase vocoder) build on this accessor.
