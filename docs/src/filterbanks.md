```@meta
CurrentModule = Audio911
```

# [Filterbanks](@id filterbanks)

A filterbank is an `nbands × nbins` matrix evaluated on the frequency grid
of a front end. Audio911 designs three kinds:

- [`auditory_fbank`](@ref): triangular filters on the `htk` mel, `slaney`
  mel or `bark` scale, in the linear (`domain=:linear`) or warped
  (`domain=:warped`) domain, normalised by `bandwidth`, `area` or
  `none_norm`. MATLAB's `designAuditoryFilterBank`; librosa's
  `filters.mel` is `scale=slaney, norm=bandwidth, domain=:linear`.
- [`gammatone_fbank`](@ref): gammatone filters equally spaced on the ERB
  scale, whose responses are evaluated directly on the grid; every bin
  strictly between DC and Nyquist is doubled (one-sided convention).
- [`chroma_fbank`](@ref): librosa's pitch-class projection.
- [`etsi_fbank`](@ref) and [`psf_fbank`](@ref): the integer-bin triangles
  of ETSI ES 201 108 and python_speech_features.

```julia
fb = auditory_fbank(16000; nfft=512, nbands=26, scale=htk, norm=bandwidth, freqrange=(100, 8000))
fb = auditory_fbank(stft; nbands=26)         # on the grid of any front end
fb = gammatone_fbank(cwt; nbands=32)
get_data(fb), get_freq(fb), get_bandwidth(fb)
```

The `sfreq` keyword takes any ascending grid; its element type fixes the
element type of the filterbank. All designs are checked against MATLAB
fixtures for FFT grids.
