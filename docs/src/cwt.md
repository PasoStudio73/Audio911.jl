```@meta
CurrentModule = Audio911
```

# [Wavelets](@id cwt)

[`Cwt`](@ref) is the wavelet alternative to the STFT. An analytic mother
wavelet ([`morlet`](@ref), [`morse`](@ref) or [`bump`](@ref), or any
function of the angular frequency) is applied scale by scale in the
frequency domain, with one inverse FFT per scale and L1 normalisation. The
scales are spaced geometrically, `voices` per octave between `freqrange`.
The squared modulus is then pooled over the frames of the `Frames` object
(weighted by its window), so the scalogram has exactly one column per frame
and can replace an STFT anywhere:

```julia
frames = Frames(audio; winsize=512, winstep=256, type=rect)
cwt    = Cwt(frames; wavelet=morse, voices=16, freqrange=(50, 8000))
mel    = MelSpec(cwt; nbands=26)
mfcc   = Mfcc(mel; ncoeffs=13)
chroma = Chroma(cwt)
```

[`get_freq`](@ref) returns the centre frequency of every scale (ascending)
and [`get_scales`](@ref) the scales in samples. A scalogram has no analysis
window, so `win_norm` on the stages fed by it is a no-op
([`get_winnorm`](@ref) returns 1). A `bump` wavelet with many voices gives a
constant-Q-like representation; librosa's CQT kernels are not reproduced.

Edge effects are handled by reflecting the signal by four times the
wavelet's time support at the lowest frequency, and the transform is
threaded over scales.
