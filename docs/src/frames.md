```@meta
CurrentModule = Audio911
```

# [Frames](@id frames)

[`Frames`](@ref) cuts a mono signal into frames of `winsize` samples every
`winstep` samples and attaches an analysis window. Frames are lazy: the
object holds the signal, the frame starts and the window, and every
consumer streams through them with one buffer ([`frame!`](@ref)).
[`get_data`](@ref) materialises the `winsize × nframes` matrix when you
want it.

```julia
frames = Frames(audio; winsize=512, winstep=256, type=hamming, periodic=true)
frames = Frames(x, sr; win=movingwindow(winsize=512, winstep=256))   # same thing
```

Options:

- `type`: `rect`, `hanning`, `hamming`, `cosine`, `lanczos`, `triang`,
  `bartlett`, `bartlett_hann`, `blackman` (from DSP) or [`povey`](@ref).
- `periodic`: MATLAB's `"periodic"` window (the symmetric window of length
  `n + 1` without its last sample) or the symmetric one.
- `center` and `pad_mode`: librosa-style centring, frame `i` centred on
  sample `(i-1)*winstep`.
- `preemph`: per-frame pre-emphasis (HTK, Kaldi, ETSI). [`preemphasis`](@ref)
  is the signal-level form.
- `dc_removal`: per-frame mean removal (Kaldi).
- `pad_end`: keep a trailing partial frame by zero padding
  (python_speech_features).

Accessors: [`get_size`](@ref), [`get_step`](@ref), [`get_overlap`](@ref),
[`get_window`](@ref), [`get_signal`](@ref), [`get_energy`](@ref) (raw
frame energies), [`get_winframes`](@ref), [`get_offset`](@ref),
[`get_times`](@ref) on any stage built from the frames.

Frames feed [`Stft`](@ref), [`Cwt`](@ref) and the time-domain descriptors
([`Rms`](@ref), [`Energy`](@ref), [`Zcr`](@ref), [`Pitch`](@ref),
[`HarmonicRatio`](@ref)).
