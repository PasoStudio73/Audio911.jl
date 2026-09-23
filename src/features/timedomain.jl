# ---------------------------------------------------------------------------- #
#                         frame-level time-domain features                     #
# ---------------------------------------------------------------------------- #
# These descriptors read the frames directly (no transform). Each stores one
# value per frame, keeps the Frames as its parent and therefore shares the
# time axis (get_times) of every spectral feature computed from the same
# frames.

struct TimeSetup <: AbstractSetup
    sr::Int64
end

struct ZcrSetup <: AbstractSetup
    sr::Int64
    threshold::Float64
    rate::Bool
end

struct PitchSetup <: AbstractSetup
    sr::Int64
    method::Base.Callable
    range::FreqRange
    threshold::Float64
end

struct HarmonicRatioSetup <: AbstractSetup
    sr::Int64
    range::FreqRange
end

# iterate the raw frames through one buffer, calling f(buf) per frame
function _map_frames(f::Function, frames::Frames{T}, ::Type{R}=T; windowed::Bool=false) where {T,R}
    n   = length(frames)
    out = Vector{R}(undef, n)
    buf = Vector{T}(undef, get_size(frames))
    w   = get_window(frames)
    @inbounds for i in 1:n
        frame!(buf, frames, i)
        windowed && (buf .*= w)
        out[i] = f(buf)
    end
    return out
end

# ---------------------------------------------------------------------------- #
#                                     rms                                      #
# ---------------------------------------------------------------------------- #
@descriptor Rms TimeSetup """
    Rms(frames::Frames; windowed=false) -> Rms

Root-mean-square level `sqrt(mean(x.^2))` of every frame (librosa `rms`).
With `windowed=true` the analysis window is applied first.
"""
Rms(frames::Frames{T}; windowed::Bool=false) where T =
    Rms{typeof(frames),T}(_map_frames(x -> sqrt(sum(abs2, x) / length(x)), frames; windowed),
                          frames, TimeSetup(get_sr(frames)))

"""
    Rms(spec::AbstractSpectrogram) -> Rms

RMS level estimated from a front-end power spectrogram through Parseval's
theorem (librosa `rms(S=...)`): `sqrt(Σ|X|² / nfft²)` with the DC and
Nyquist bins counted once and the others twice.
"""
function Rms(spec::AbstractSpectrogram)
    T = eltype(spec)
    S = get_spectrum(spec) === power ? get_spec(spec) : get_spec(spec) .^ 2
    f = get_freq(spec)
    nyq = T(get_sr(spec)) / 2
    nb, nf = size(S)
    nfft = T(2 * (nb - 1))
    r = Vector{T}(undef, nf)
    @inbounds for j in 1:nf
        acc = zero(T)
        for k in 1:nb
            g = (f[k] > 0 && f[k] < nyq) ? T(2) : one(T)
            acc += g * S[k, j]
        end
        r[j] = sqrt(acc) / nfft
    end
    return Rms{typeof(spec),T}(r, spec, TimeSetup(get_sr(spec)))
end

# ---------------------------------------------------------------------------- #
#                                    energy                                    #
# ---------------------------------------------------------------------------- #
@descriptor Energy TimeSetup """
    Energy(frames::Frames; windowed=true) -> Energy

Short-time energy `Σ x.^2` of every windowed frame (MATLAB
`shortTimeEnergy`). `windowed=false` gives the raw frame energy, the same
values as [`get_energy`](@ref).
"""
Energy(frames::Frames{T}; windowed::Bool=true) where T =
    Energy{typeof(frames),T}(_map_frames(x -> sum(abs2, x), frames; windowed),
                             frames, TimeSetup(get_sr(frames)))

# ---------------------------------------------------------------------------- #
#                              zero-crossing rate                              #
# ---------------------------------------------------------------------------- #
function _zcr(x::AbstractVector{T}, threshold::T) where T
    c = 0
    prev = x[1]
    @inbounds for i in 2:length(x)
        v = x[i]
        # samples inside (-threshold, threshold) are treated as zero (librosa)
        a = abs(prev) <= threshold ? zero(T) : prev
        b = abs(v) <= threshold ? zero(T) : v
        (signbit(a) != signbit(b)) && (c += 1)
        prev = v
    end
    return c
end

@descriptor Zcr ZcrSetup """
    Zcr(frames::Frames; threshold=1e-10, rate=true) -> Zcr

Zero-crossing rate of every frame: number of sign changes divided by the
frame length (librosa `zero_crossing_rate`, MATLAB `zerocrossrate`).
Samples with `|x| ≤ threshold` count as zero. `rate=false` returns the
crossing counts instead; `windowed=true` counts on the windowed frames and
`strict=true` counts only strict sign changes `x[i] x[i-1] < 0`, so exact
zeros never cross (both are audioFlux `Temporal`'s rules).
"""
function Zcr(frames::Frames{T}; threshold::Real=1e-10, rate::Bool=true, windowed::Bool=false,
             strict::Bool=false) where T
    th = T(threshold)
    n  = T(get_size(frames))
    count_zc(x) = strict ? count(i -> x[i] * x[i - 1] < 0, 2:length(x)) : _zcr(x, th)
    v  = _map_frames(x -> rate ? T(count_zc(x)) / n : T(count_zc(x)), frames; windowed)
    return Zcr{typeof(frames),T}(v, frames, ZcrSetup(get_sr(frames), Float64(threshold), rate))
end

"""
    zero_crossings(x; threshold=1e-10) -> Vector{Bool}

`true` at every sample where the sign changes with respect to the previous
sample (librosa `zero_crossings`); the first element is `false`.
"""
function zero_crossings(x::AbstractVector{T}; threshold::Real=1e-10) where {T<:Real}
    th = T(threshold)
    z = falses(length(x))
    @inbounds for i in 2:length(x)
        a = abs(x[i-1]) <= th ? zero(T) : x[i-1]
        b = abs(x[i]) <= th ? zero(T) : x[i]
        z[i] = signbit(a) != signbit(b)
    end
    return z
end

# ---------------------------------------------------------------------------- #
#                                autocorrelation                               #
# ---------------------------------------------------------------------------- #
"""
    autocorrelate(x; max_size=length(x)) -> Vector

Bounded auto-correlation `r[k] = Σ x[n] x[n+k]` for `k = 0:max_size-1`,
computed with an FFT (librosa `autocorrelate`).
"""
function autocorrelate(x::AbstractVector{T}; max_size::Int=length(x)) where {T<:Real}
    n = length(x)
    m = nextpow(2, 2n)
    X = rfft(vcat(Vector{T}(x), zeros(T, m - n)))
    r = irfft(abs2.(X), m)
    return r[1:min(max_size, n)]
end

# ---------------------------------------------------------------------------- #
#                                     pitch                                    #
# ---------------------------------------------------------------------------- #
# lag range (in samples) for a frequency range
function _lag_range(sr::Int, range::FreqRange, n::Int)
    kmin = max(1, floor(Int, sr / get_hi(range)))
    kmax = min(n - 1, ceil(Int, sr / get_low(range)))
    kmin < kmax || throw(ArgumentError("pitch range $range is not resolvable with $n-sample frames at $sr Hz"))
    return kmin, kmax
end

# parabolic interpolation of a peak at index k of v
@inline function _parabolic(v, k)
    (k ≤ 1 || k ≥ length(v)) && return float(k)
    a, b, c = v[k-1], v[k], v[k+1]
    d = a - 2b + c
    return d == 0 ? float(k) : k + (a - c) / (2d)
end

"""
    pitch_ncf(x, sr; range=(50, 400), threshold=0) -> f0

Normalised correlation function estimate for one frame (MATLAB `pitch`
`Method="NCF"`): `r[k] / sqrt(r[0]·e[k])` with `e[k]` the energy of the
lagged segment, maximised over the lags of `range`. Returns `0` when the
best correlation is below `threshold`.
"""
function pitch_ncf(x::AbstractVector{T}, sr::Int; range::FreqRange=(50, 400), threshold::Real=0) where T
    n = length(x)
    kmin, kmax = _lag_range(sr, range, n)
    r = autocorrelate(x; max_size=kmax + 1)
    e0 = r[1]
    e0 > 0 || return zero(T)
    # energy of x[k+1:end] for every lag, via a running sum
    best, bestk = -Inf, 0
    tail = sum(abs2, x)
    for k in 1:kmax
        tail -= x[k]^2
        k < kmin && continue
        val = r[k + 1] / sqrt(e0 * max(tail, floatmin(T)))
        if val > best
            best, bestk = val, k
        end
    end
    (bestk == 0 || best < threshold) && return zero(T)
    ncf = [k ≤ kmax ? r[k + 1] : zero(T) for k in 0:kmax]
    lag = _parabolic(ncf, bestk + 1) - 1
    return T(sr / lag)
end

"""
    pitch_yin(x, sr; range=(50, 400), threshold=0.1) -> f0

YIN estimate for one frame (de Cheveigné & Kawahara 2002, librosa `yin`):
cumulative-mean-normalised difference function, first dip below
`threshold` (else the global minimum), parabolic interpolation. Returns `0`
when no dip exists.
"""
function pitch_yin(x::AbstractVector{T}, sr::Int; range::FreqRange=(50, 400), threshold::Real=0.1) where T
    n = length(x)
    kmin, kmax = _lag_range(sr, range, n ÷ 2 + 1)
    W = n - kmax
    W > 0 || throw(ArgumentError("frame too short for the pitch range $range"))
    # difference function d[k] = Σ_{i=1}^{W} (x[i] - x[i+k])^2 for k = 0..kmax
    d = Vector{T}(undef, kmax + 1)
    @inbounds for k in 0:kmax
        acc = zero(T)
        @simd for i in 1:W
            acc += (x[i] - x[i + k])^2
        end
        d[k + 1] = acc
    end
    # cumulative mean normalised difference
    cmnd = similar(d)
    cmnd[1] = one(T)
    run = zero(T)
    @inbounds for k in 1:kmax
        run += d[k + 1]
        cmnd[k + 1] = run > 0 ? d[k + 1] * k / run : one(T)
    end
    th = T(threshold)
    k = kmin
    found = 0
    @inbounds while k ≤ kmax
        if cmnd[k + 1] < th
            while k + 1 ≤ kmax && cmnd[k + 2] < cmnd[k + 1]
                k += 1
            end
            found = k
            break
        end
        k += 1
    end
    if found == 0
        seg = view(cmnd, kmin+1:kmax+1)
        found = kmin + argmin(seg) - 1
    end
    lag = _parabolic(cmnd, found + 1) - 1
    lag > 0 || return zero(T)
    return T(sr / lag)
end

"""
    pitch_cep(x, sr; range=(50, 400), threshold=0) -> f0

Cepstral estimate for one frame (MATLAB `pitch` `Method="CEP"`): the real
cepstrum of the frame is searched for its highest peak between the
quefrencies of `range`.
"""
function pitch_cep(x::AbstractVector{T}, sr::Int; range::FreqRange=(50, 400), threshold::Real=0) where T
    n = length(x)
    m = nextpow(2, 2n)
    X = rfft(vcat(Vector{T}(x), zeros(T, m - n)))
    c = irfft(log.(abs.(X) .+ eps(T)), m)
    kmin, kmax = _lag_range(sr, range, m ÷ 2)
    seg = view(c, kmin+1:kmax+1)
    k = kmin + argmax(seg) - 1
    c[k + 1] > threshold || return zero(T)
    return T(sr / (_parabolic(c, k + 1) - 1))
end

@descriptor Pitch PitchSetup """
    Pitch(frames::Frames; method=pitch_ncf, range=(50, 400), threshold=...) -> Pitch

Fundamental-frequency estimate of every frame in Hz, `0` when unvoiced.
`method` is one of [`pitch_ncf`](@ref) (MATLAB's default), [`pitch_yin`](@ref)
(librosa `yin`) or [`pitch_cep`](@ref); `range` bounds the search and
`threshold` is passed to the method (defaults `0`, `0.1`, `0` respectively).

```julia
f0 = Pitch(Frames(audio; winsize=1024, winstep=256); method=pitch_yin, range=(60, 500))
get_data(f0), get_times(f0)
```
"""
function Pitch(frames::Frames{T}; method::Base.Callable=pitch_ncf, range::FreqRange=(50, 400),
               threshold::Maybe{Real}=nothing) where T
    th = isnothing(threshold) ? (method === pitch_yin ? 0.1 : 0.0) : Float64(threshold)
    v  = _map_frames(x -> method(x, get_sr(frames); range, threshold=th), frames)
    return Pitch{typeof(frames),T}(v, frames, PitchSetup(get_sr(frames), method, range, th))
end

# ---------------------------------------------------------------------------- #
#                                harmonic ratio                                #
# ---------------------------------------------------------------------------- #
@descriptor HarmonicRatio HarmonicRatioSetup """
    HarmonicRatio(frames::Frames; range=(50, 400), method=:matlab) -> HarmonicRatio

Maximum of the normalised auto-correlation of every frame over the lags of
`range` (MATLAB `harmonicRatio`); 1 for a perfectly periodic frame, near 0
for noise.

`method=:audioflux` computes audioFlux's `HarmonicRatio` instead: the
windowed frame is zero-padded to twice its length for the auto-correlation,
the lag search starts at the first zero crossing of the auto-correlation
(kept from the previous frame when there is none) and runs up to
`sr / fmin` (`fmin` defaults to `range[1]`), the lag energy is the cumulative energy of the frame
(audioFlux's index, one sample shorter than the overlap) and the maximum is
refined by quadratic interpolation. Build the frames with audioFlux's
Hamming window to match it.
"""
function HarmonicRatio(frames::Frames{T}; range::FreqRange=(50, 400), method::Symbol=:matlab,
                       fmin::Real=get_low(range)) where T
    method in (:matlab, :audioflux) || throw(ArgumentError("method must be :matlab or :audioflux, got :$method"))
    method === :audioflux && return HarmonicRatio{typeof(frames),T}(_hr_audioflux(frames, fmin),
                                                                     frames, HarmonicRatioSetup(get_sr(frames), range))
    sr = get_sr(frames)
    v  = _map_frames(frames) do x
        n = length(x)
        kmin, kmax = _lag_range(sr, range, n)
        r  = autocorrelate(x; max_size=kmax + 1)
        e0 = r[1]
        e0 > 0 || return zero(T)
        best = zero(T)
        tail = e0
        for k in 1:kmax
            tail -= x[k]^2
            k < kmin && continue
            best = max(best, r[k + 1] / sqrt(e0 * max(tail, floatmin(T))))
        end
        return best
    end
    return HarmonicRatio{typeof(frames),T}(v, frames, HarmonicRatioSetup(sr, range))
end

# audioFlux's harmonic ratio (src/mir/harmonicRatio_algorithm.c)
function _hr_audioflux(frames::Frames{T}, fmin::Real) where T
    sr = get_sr(frames)
    W  = get_winsize(frames)
    M  = 2W
    maxL = min(floor(Int, sr / fmin), W - 1)
    w  = get_window(frames)
    plan = plan_rfft(zeros(T, M))
    buf  = zeros(T, M)
    X    = Vector{Complex{T}}(undef, M ÷ 2 + 1)
    raw  = Vector{T}(undef, W)
    out  = Vector{T}(undef, length(frames))
    minidx = 0                                    # kept across frames, as in audioFlux
    for j in 1:length(frames)
        frame!(raw, frames, j)
        @inbounds for i in 1:W; buf[i] = raw[i] * w[i]; end
        @inbounds fill!(view(buf, W+1:M), zero(T))
        mul!(X, plan, buf)
        r  = irfft(abs2.(X), M)                   # r[k+1]: lag k
        cs = cumsum(view(buf, 1:W) .^ 2)
        for k in 2:maxL
            if (r[k + 1] ≥ 0 && r[k] ≤ 0) || (r[k + 1] ≤ 0 && r[k] ≥ 0)
                minidx = k - 1
                break
            end
        end
        n = maxL - minidx - 1
        if n ≤ 0
            out[j] = zero(T); continue
        end
        g = T[r[k + 1] / sqrt(r[1] * cs[W - 1 - k] + T(1e-16)) for k in minidx+1:maxL-1]
        i = argmax(g)
        if i == 1 || i == n
            out[j] = g[i]
        else
            v1, v2, v3 = g[i - 1], g[i], g[i + 1]
            p = (v3 - v1) / (2 * (2v2 - v3 - v1) + T(1e-16))
            out[j] = v2 - T(0.25) * (v1 - v3) * p
        end
    end
    return out
end
