# ---------------------------------------------------------------------------- #
#                   wavelet synchrosqueezing (synsq and wsst)                  #
# ---------------------------------------------------------------------------- #
# Ports of audioFlux's synsq and wsst (src/synsq_algorithm.c,
# src/wsst_algorithm.c, MIT licence, Copyright (c) 2023 libAudioFlux), after
# Daubechies, Lu & Wu (2011): every wavelet coefficient is moved, at its own
# time, to the band nearest to its instantaneous frequency. `synsq` estimates
# the instantaneous frequency from the phase difference of consecutive
# coefficients, `wsst` from the coefficients of the derivative wavelet
# (Im(W_∂ / W) / 2π). audioFlux maps frequencies on the octave, log and
# linear grids with `num / (num - 1)` times the bin step; here every grid
# maps to its nearest band.

# the grid is geometric (octave, logspace, voices) or not
_log_grid(scale) = isnothing(scale) || scale === octave || scale === logspace

# nearest band of frequency `f` on the ascending grid `freq`, 0 when outside:
# geometric grids compare in log2 and accept half a step beyond the ends,
# other grids accept [freq[1], freq[end]) (audioFlux's nearest-band rule)
function _band_index(freq::AbstractVector, f::Real, logd::Bool)
    n = length(freq)
    (isfinite(f) && f > 0) || return 0
    if logd
        l1, ln = log2(freq[1]), log2(freq[n])
        step = n > 1 ? (ln - l1) / (n - 1) : one(l1)
        k = round(Int, (log2(f) - l1) / step) + 1
        return 1 ≤ k ≤ n ? k : 0
    end
    (freq[1] ≤ f < freq[n]) || return 0
    i = searchsortedlast(freq, f)
    return (f - freq[i]) < (freq[i + 1] - f) ? i : i + 1
end

# repeat the band map `order - 1` times (target of the target, same time)
function _reorder!(idx::AbstractMatrix{<:Integer}, order::Int)
    order ≤ 1 && return idx
    tmp = similar(idx)
    for _ in 2:order
        @inbounds for n in axes(idx, 2), k in axes(idx, 1)
            t = idx[k, n]
            tmp[k, n] = t > 0 ? idx[t, n] : 0
        end
        idx .= tmp
    end
    return idx
end

# instantaneous frequency (Hz) of a band series from the phase difference of
# consecutive coefficients (audioFlux synsq: 0 at the first sample, the last
# sample repeats the one before)
function _phase_freq!(out::AbstractVector{T}, y::AbstractVector{<:Complex}, sr::Int) where T
    N = length(y)
    out[1] = zero(T)
    @inbounds for n in 2:N
        out[n] = abs(rem2pi(angle(y[n]) - angle(y[n - 1]), RoundNearest)) * T(sr) / T(2π)
    end
    N ≥ 3 && (out[N] = out[N - 1])
    return out
end

"""
    synsq(W, freq, sr; scale=octave, thresh=0.001, order=1) -> Matrix{Complex}

Synchrosqueezing of a complex wavelet transform `W` (`bands × samples`, rows
on the ascending grid `freq`, e.g. from [`cwt`](@ref)); audioFlux `Synsq`.
The instantaneous frequency of every coefficient is the phase difference
to the previous sample over `2π`; the coefficient is added to the band
nearest to it at the same sample (log-domain nearest band on the `octave`
and `logspace` grids, `scale=nothing` for a geometric grid, linear
otherwise). Coefficients with `|W| ≤ thresh` or a frequency outside the grid
are dropped; `order > 1` repeats the band mapping.
"""
function synsq(W::AbstractMatrix{Complex{T}}, freq::AbstractVector, sr::Int;
               scale::Maybe{Base.Callable}=octave, thresh::Real=0.001, order::Int=1) where T
    nb, N = size(W)
    length(freq) == nb || throw(DimensionMismatch("W has $nb rows, freq has $(length(freq)) values"))
    order ≥ 1 || throw(ArgumentError("order must be ≥ 1"))
    logd = _log_grid(scale)
    idx = zeros(Int32, nb, N)
    f = Vector{T}(undef, N)
    for k in 1:nb
        _phase_freq!(f, view(W, k, :), sr)
        @inbounds for n in 1:N
            idx[k, n] = _band_index(freq, f[n], logd)
        end
    end
    return _squeeze(W, idx, thresh, order)
end

function _squeeze(W::AbstractMatrix{Complex{T}}, idx::AbstractMatrix{<:Integer}, thresh::Real, order::Int) where T
    _reorder!(idx, order)
    out = zeros(Complex{T}, size(W))
    th2 = T(thresh)^2
    @inbounds for n in axes(W, 2), k in axes(W, 1)
        t = idx[k, n]
        (t > 0 && abs2(W[k, n]) > th2) && (out[t, n] += W[k, n])
    end
    return out
end

"""
    wsst(x, sr; wavelet=morse, nbands=84, scale=octave, freqrange, bins_per_octave=12,
         centre, pad=true, thresh=0.001, order=1) -> (S, W, freq)

Wavelet synchrosqueezed transform of a whole signal (audioFlux `WSST`): the
complex CWT `W` (see [`cwt`](@ref)) and its time derivative `W_∂` (the
wavelet multiplied by `iω`) give the instantaneous frequency
`|Im(W_∂ / W)| · sr / 2π` of every coefficient, which is added to the
nearest band at the same sample. Returns the squeezed transform `S`, the
CWT `W` and the band frequencies.
"""
function wsst(x::AbstractVector{<:Real}, sr::Int; wavelet::Base.Callable=morse, nbands::Int=84,
              scale::Maybe{Base.Callable}=octave, voices::Int=12,
              freqrange::FreqRange=(_log_grid(scale) ? 33 : 0, sr ÷ 2),
              bins_per_octave::Int=12, centre::Real=_centre_omega(wavelet), pad::Bool=true,
              thresh::Real=0.001, order::Int=1, T::Type=float(eltype(x)))
    order ≥ 1 || throw(ArgumentError("order must be ≥ 1"))
    N = length(x)
    freq = _cwt_grid(T, scale, freqrange, nbands, voices, bins_per_octave)
    p  = pad ? N ÷ 2 : 0
    xp = pad ? _symmetric_pad(Vector{T}(x), p) : Vector{T}(x)
    W  = Matrix{Complex{T}}(undef, length(freq), N)
    idx = zeros(Int32, length(freq), N)
    logd = _log_grid(scale)
    _cwt_bands(xp, p, N, freq, sr, wavelet, centre; deriv=true) do _, k, w, dw
        @inbounds for n in 1:N
            W[k, n] = w[n]
            f = abs(imag(dw[n] / w[n])) * T(sr) / T(2π)
            idx[k, n] = _band_index(freq, f, logd)
        end
    end
    return _squeeze(W, idx, thresh, order), W, freq
end

# ---------------------------------------------------------------------------- #
#                                  front ends                                  #
# ---------------------------------------------------------------------------- #
struct SqueezeSetup <: AbstractSetup
    sr         :: Int64
    method     :: Symbol          # :synsq or :wsst
    thresh     :: Float64
    order      :: Int64
    accumulate :: Symbol
    spectrum   :: Base.Callable
end

"""
    Synchrosqueezed{F,T} <: AbstractSpectrogram

A synchrosqueezed wavelet spectrogram on the grid of a [`Cwt`](@ref) (same
bands, same frames), built by [`Synsq`](@ref) or [`Wsst`](@ref). It
implements the front-end interface.
"""
struct Synchrosqueezed{F,T<:AudioData} <: AbstractSpectrogram
    spec   :: Matrix{T}
    parent :: F
    info   :: SqueezeSetup
end

Base.eltype(::Synchrosqueezed{F,T}) where {F,T} = T
get_data(s::Synchrosqueezed)     = s.spec
get_spec(s::Synchrosqueezed)     = s.spec
get_freq(s::Synchrosqueezed)     = get_freq(s.parent)
get_setup(s::Synchrosqueezed)    = s.info
get_sr(s::Synchrosqueezed)       = s.info.sr
get_spectrum(s::Synchrosqueezed) = s.info.spectrum
get_parent(s::Synchrosqueezed)   = s.parent
get_frontend(s::Synchrosqueezed) = s
Base.show(io::IO, s::Synchrosqueezed{F,T}) where {F,T} =
    print(io, "Synchrosqueezed{$T}(:$(s.info.method), $(size(s.spec, 2)) frames × $(size(s.spec, 1)) bands)")

# squeeze a Cwt's bands and pool them on its frames
function _squeeze_pooled(c::Cwt{T}, method::Symbol; thresh::Real, order::Int, accumulate::Symbol,
                         spectrum::Base.Callable) where T
    order ≥ 1 || throw(ArgumentError("order must be ≥ 1"))
    thresh ≥ 0 || throw(ArgumentError("thresh must be ≥ 0"))
    accumulate in (:energy, :complex) || throw(ArgumentError("accumulate must be :energy or :complex, got :$accumulate"))
    _check_spectrum(spectrum)
    info   = c.info
    frames = c.frames
    freq   = c.freq
    sr     = info.sr
    x      = get_signal(frames)
    N      = length(x)
    nb     = length(freq)
    nf     = length(frames)
    xp, pad = _reflect_pad(x, info.pad)
    logd   = _log_grid(info.scale)
    deriv  = method === :wsst
    th2    = T(thresh)^2
    w      = get_window(frames)
    ws     = length(w)
    wsum   = sum(w)

    # band index of every coefficient; needed up front for order > 1 or a
    # complex accumulation, otherwise computed on the fly
    need_map = order > 1 || accumulate === :complex
    idx = need_map ? zeros(Int32, nb, N) : nothing
    target!(t, k, y, dy) = begin
        if deriv
            @inbounds for n in 1:N
                t[n] = _band_index(freq, abs(imag(dy[n] / y[n])) * T(sr) / T(2π), logd)
            end
        else
            f = Vector{T}(undef, N)
            _phase_freq!(f, y, sr)
            @inbounds for n in 1:N
                t[n] = _band_index(freq, f[n], logd)
            end
        end
        t
    end
    if need_map
        _cwt_bands(xp, pad, N, freq, sr, info.wavelet, info.centre; deriv) do _, k, y, dy
            target!(view(idx, k, :), k, y, dy)
        end
        _reorder!(idx, order)
    end

    if accumulate === :complex
        # coefficients summed in their target band at every sample, then pooled
        A = zeros(Complex{T}, nb, N)
        lk = ReentrantLock()
        _cwt_bands(xp, pad, N, freq, sr, info.wavelet, info.centre) do _, k, y, _
            lock(lk) do
                @inbounds for n in 1:N
                    t = idx[k, n]
                    (t > 0 && abs2(y[n]) > th2) && (A[t, n] += y[n])
                end
            end
        end
        spec = Matrix{T}(undef, nb, nf)
        for k in 1:nb
            _pool_band!(spec, k, view(A, k, :), frames, spectrum)
        end
    else
        # energies added directly into the pooled output, one accumulator per thread chunk
        nchunks = length(_chunks(nb))
        accs = [zeros(T, nb, nf) for _ in 1:nchunks]
        _cwt_bands(xp, pad, N, freq, sr, info.wavelet, info.centre; deriv=deriv && !need_map) do c, k, y, dy
            t = need_map ? view(idx, k, :) : target!(Vector{Int32}(undef, N), k, y, dy)
            acc = accs[c]
            @inbounds for (j, st) in enumerate(frames.starts)
                base = st - 1
                for i in 1:ws
                    n = base + i
                    tk = t[n]
                    (tk > 0 && abs2(y[n]) > th2) || continue
                    acc[tk, j] += w[i] * T(spectrum(y[n])) / wsum
                end
            end
        end
        spec = reduce(+, accs)
    end
    return Synchrosqueezed{typeof(c),T}(spec, c, SqueezeSetup(sr, method, Float64(thresh), order, accumulate, spectrum))
end

"""
    Synsq(c::Cwt; thresh=0.001, order=1, accumulate=:energy, spectrum=get_spectrum(c)) -> Synchrosqueezed

Synchrosqueezed scalogram of a [`Cwt`](@ref) (audioFlux `synsq`): the
instantaneous frequency of every wavelet coefficient is estimated from the
phase difference to the previous sample and the coefficient is moved to the
nearest band at the same time, then pooled on the frames of `c`.
`accumulate=:energy` adds the power (or magnitude) of the moved
coefficients, keeping memory at the size of the output; `:complex` adds the
complex coefficients first, as audioFlux does, at the cost of a
`bands × samples` complex buffer. See [`synsq`](@ref) for the whole-signal
form.
"""
Synsq(c::Cwt; thresh::Real=0.001, order::Int=1, accumulate::Symbol=:energy,
      spectrum::Base.Callable=get_spectrum(c)) =
    _squeeze_pooled(c, :synsq; thresh, order, accumulate, spectrum)

"""
    Wsst(c::Cwt; thresh=0.001, order=1, accumulate=:energy, spectrum=get_spectrum(c)) -> Synchrosqueezed
    Wsst(frames::Frames; thresh, order, accumulate, kwargs...) -> Synchrosqueezed

Wavelet synchrosqueezed transform (Daubechies, Lu & Wu 2011; audioFlux
`wsst`) on the grid of a [`Cwt`](@ref): the instantaneous frequency comes
from the derivative wavelet, `|Im(W_∂/W)| · sr/2π`. The second form builds
the `Cwt` from `frames` with the remaining keywords (defaults: Morse
wavelet, octave grid from C1). See [`wsst`](@ref) for the whole-signal form.

```julia
frames = Frames(audio; winsize=512, winstep=256, type=rect)
w = Wsst(frames; scale=octave, nbands=84, freqrange=(33, 8000))
mel = MelSpec(w; nbands=20, freqrange=(50, 3900))
```
"""
Wsst(c::Cwt; thresh::Real=0.001, order::Int=1, accumulate::Symbol=:energy,
     spectrum::Base.Callable=get_spectrum(c)) =
    _squeeze_pooled(c, :wsst; thresh, order, accumulate, spectrum)

Wsst(frames::Frames; thresh::Real=0.001, order::Int=1, accumulate::Symbol=:energy,
     wavelet::Base.Callable=morse, scale::Maybe{Base.Callable}=octave, nbands::Int=84,
     freqrange::FreqRange=(33, get_sr(frames) ÷ 2), kwargs...) =
    Wsst(Cwt(frames; wavelet, scale, nbands, freqrange, kwargs...); thresh, order, accumulate)
