# ---------------------------------------------------------------------------- #
#                      derived spectrograms on the same grid                   #
# ---------------------------------------------------------------------------- #
struct DerivedSetup <: AbstractSetup
    sr::Int64
    name::Symbol
end

"""
    DerivedSpec{F,T} <: AbstractSpectrogram

A spectrogram derived from another one on the same frequency grid (a masked
component, a gated or PCEN-compressed version). It implements the full
front-end interface, so it can feed filterbanks, cepstra and descriptors
exactly like the spectrogram it came from.
"""
struct DerivedSpec{F,T<:AudioData} <: AbstractSpectrogram
    spec   :: Matrix{T}
    parent :: F
    info   :: DerivedSetup
end

Base.eltype(::DerivedSpec{F,T}) where {F,T} = T
get_data(d::DerivedSpec)     = d.spec
get_spec(d::DerivedSpec)     = d.spec
get_freq(d::DerivedSpec)     = get_freq(d.parent)
get_setup(d::DerivedSpec)    = d.info
get_sr(d::DerivedSpec)       = d.info.sr
get_parent(d::DerivedSpec)   = d.parent
get_window(d::DerivedSpec)   = get_window(d.parent)
get_nfft(d::DerivedSpec)     = get_nfft(d.parent)
"""
    get_name(d::DerivedSpec) -> Symbol

What the derived spectrogram is (`:harmonic`, `:percussive`, `:gated`, `:pcen`).
"""
get_name(d::DerivedSpec)     = d.info.name
get_frontend(d::DerivedSpec) = d
_freq_indices(d::DerivedSpec, fr::FreqRange) = _freq_indices(d.parent, fr)
Base.show(io::IO, d::DerivedSpec{F,T}) where {F,T} =
    print(io, "DerivedSpec{$(nameof(F)),$T}(:$(d.info.name), $(size(d.spec, 2)) frames × $(size(d.spec, 1)) bins)")

_derived(s::AbstractSpectrogram, spec::Matrix, name::Symbol) =
    DerivedSpec{typeof(s),eltype(s)}(spec, s, DerivedSetup(get_sr(s), name))

# ---------------------------------------------------------------------------- #
#                                median filters                                #
# ---------------------------------------------------------------------------- #
# median over a centred window of `k` samples along dim 2 (frames) or dim 1
# (bins); at the edges the window shrinks (`zero_pad=false`) or the matrix is
# padded with zeros (`zero_pad=true`, scipy `medfilt`, audioFlux)
function _medfilt(S::AbstractMatrix{T}, k::Int, dim::Int; zero_pad::Bool=false) where T
    k ≤ 1 && return copy(S)
    nb, nf = size(S)
    h = k ÷ 2
    R = similar(S)
    buf = Vector{T}(undef, k)
    if dim == 2
        @inbounds for i in 1:nb, j in 1:nf
            lo, hi = max(1, j - h), min(nf, j + (k - 1 - h))
            m = hi - lo + 1
            copyto!(buf, 1, view(S, i, lo:hi), 1, m)
            zero_pad && (fill!(view(buf, m+1:k), zero(T)); m = k)
            R[i, j] = _median!(view(buf, 1:m))
        end
    else
        @inbounds for j in 1:nf, i in 1:nb
            lo, hi = max(1, i - h), min(nb, i + (k - 1 - h))
            m = hi - lo + 1
            copyto!(buf, 1, view(S, lo:hi, j), 1, m)
            zero_pad && (fill!(view(buf, m+1:k), zero(T)); m = k)
            R[i, j] = _median!(view(buf, 1:m))
        end
    end
    return R
end

function _median!(v::AbstractVector{T}) where T
    sort!(v)
    n = length(v)
    return isodd(n) ? v[(n + 1) ÷ 2] : (v[n ÷ 2] + v[n ÷ 2 + 1]) / 2
end

# soft mask X^p / (X^p + R^p), computed stably (librosa `softmask`)
function _softmask(X::AbstractMatrix{T}, R::AbstractMatrix{T}, p::Real) where T
    M = similar(X)
    @inbounds for i in eachindex(X)
        z = max(X[i], R[i])
        if z ≤ 0
            M[i] = T(0.5)
        elseif isinf(p)
            M[i] = X[i] > R[i] ? one(T) : (X[i] == R[i] ? T(0.5) : zero(T))
        else
            a = (X[i] / z)^p; b = (R[i] / z)^p
            M[i] = a / (a + b)
        end
    end
    return M
end

# ---------------------------------------------------------------------------- #
#                                     hpss                                     #
# ---------------------------------------------------------------------------- #
"""
    Hpss{H,P,T}

Result of [`Hpss`](@ref Hpss(::AbstractSpectrogram)): the harmonic and
percussive components (`get_harmonic`, `get_percussive`) as
[`DerivedSpec`](@ref) spectrograms, plus the two soft masks.
"""
struct Hpss{H,P,T<:AudioData}
    harmonic   :: H
    percussive :: P
    mask_h     :: Matrix{T}
    mask_p     :: Matrix{T}
end

"""
    get_harmonic(h::Hpss) -> DerivedSpec

The harmonic component of an [`Hpss`](@ref) decomposition.
"""
get_harmonic(h::Hpss)   = h.harmonic

"""
    get_percussive(h::Hpss) -> DerivedSpec

The percussive component of an [`Hpss`](@ref) decomposition.
"""
get_percussive(h::Hpss) = h.percussive

"""
    get_masks(h::Hpss) -> (mask_h, mask_p)

The soft masks applied to the input spectrogram.
"""
get_masks(h::Hpss)      = (h.mask_h, h.mask_p)
Base.show(io::IO, h::Hpss) = print(io, "Hpss(", h.harmonic, ", ", h.percussive, ")")

"""
    Hpss(spec; kernel=(31, 31), power=2.0, margin=(1, 1), edge=:shrink) -> Hpss

Harmonic/percussive source separation by median filtering (Fitzgerald
2010; librosa `decompose.hpss`, audioFlux `HPSS`). The magnitude
spectrogram is median filtered along time (`kernel[1]` frames, harmonic)
and along frequency (`kernel[2]` bins, percussive); Wiener-like soft masks
with exponent `power` (`Inf` for hard masks) and separation `margin`s are
applied to the input spectrogram. Both components keep the frequency grid
and spectrum kind of `spec`. At the edges of the spectrogram the median
window shrinks (`edge=:shrink`) or the spectrogram is padded with zeros
(`edge=:zero`, audioFlux and scipy `medfilt`).

On an [`Stft`](@ref), [`get_harmonic_signal`](@ref) and
[`get_percussive_signal`](@ref) return the separated signals. audioFlux's
`HPSS(radix2_exp=11, window_type=HAMM, h_order=21, p_order=31)` is

```julia
stft = Stft(Frames(audio; winsize=2048, winstep=512, type=hamming))
h = Hpss(stft; kernel=(21, 31), edge=:zero)
yh, yp = get_harmonic_signal(h), get_percussive_signal(h)
```

```julia
h = Hpss(stft)
mel_h = MelSpec(get_harmonic(h); nbands=40)
onset = OnsetStrength(MelSpec(get_percussive(h)))
```
"""
function Hpss(s::AbstractSpectrogram; kernel::Tuple{Int,Int}=(31, 31), power::Real=2.0,
              margin::Union{Real,Tuple{Real,Real}}=(1, 1), edge::Symbol=:shrink)
    T = eltype(s)
    mh, mp = margin isa Tuple ? (T(margin[1]), T(margin[2])) : (T(margin), T(margin))
    (mh ≥ 1 && mp ≥ 1) || throw(ArgumentError("margins must be ≥ 1"))
    edge in (:shrink, :zero) || throw(ArgumentError("edge must be :shrink or :zero, got :$edge"))
    S = Matrix{T}(_magnitude_spec(s))
    H = _medfilt(S, kernel[1], 2; zero_pad=edge === :zero)
    P = _medfilt(S, kernel[2], 1; zero_pad=edge === :zero)
    maskh = _softmask(H, P .* mp, power)
    maskp = _softmask(P, H .* mh, power)
    X = get_spec(s)
    return Hpss(_derived(s, X .* maskh, :harmonic), _derived(s, X .* maskp, :percussive), maskh, maskp)
end

function _masked_signal(h::Hpss, M::AbstractMatrix; method::Symbol, length)
    s = get_parent(h.harmonic)
    s isa Stft || throw(ArgumentError("the separated signals need an Hpss of an Stft, got $(nameof(typeof(s)))"))
    return _istft(s, get_complex(s) .* M; method, length)
end

"""
    get_harmonic_signal(h::Hpss; method=:wola, length=nothing) -> Vector

The harmonic signal of an [`Hpss`](@ref) of an [`Stft`](@ref): the complex
STFT times the harmonic mask, inverted with [`istft`](@ref) (`method`,
`length` as there; by default the length of the analysed signal).
"""
get_harmonic_signal(h::Hpss; method::Symbol=:wola, length::Maybe{Int}=nothing) =
    _masked_signal(h, h.mask_h; method, length)

"""
    get_percussive_signal(h::Hpss; method=:wola, length=nothing) -> Vector

The percussive signal of an [`Hpss`](@ref) of an [`Stft`](@ref), as
[`get_harmonic_signal`](@ref) with the percussive mask.
"""
get_percussive_signal(h::Hpss; method::Symbol=:wola, length::Maybe{Int}=nothing) =
    _masked_signal(h, h.mask_p; method, length)

# ---------------------------------------------------------------------------- #
#                                  noise gates                                 #
# ---------------------------------------------------------------------------- #
"""
    noisegate(x, sr; threshold=-10, attack=0.05, release=0.2, hold=0.05) -> Vector

Time-domain noise gate after MATLAB's `noiseGate`: samples whose level
`20 log10 |x|` is below `threshold` (dB) close the gate. The gain moves
from 0 to 1 with an `attack` time and back with a `release` time (seconds,
10 %–90 % constants `exp(-log(9) / (sr * t))`), and the gate is held open
`hold` seconds after the level drops below the threshold.
"""
function noisegate(x::AbstractVector{T}, sr::Int; threshold::Real=-10, attack::Real=0.05,
                   release::Real=0.2, hold::Real=0.05) where {T<:Real}
    F = float(T)
    th  = F(10)^(F(threshold) / 20)           # linear threshold
    αa  = attack  > 0 ? F(exp(-log(9) / (sr * attack)))  : zero(F)
    αr  = release > 0 ? F(exp(-log(9) / (sr * release))) : zero(F)
    nh  = round(Int, hold * sr)
    y = similar(x, F)
    g = zero(F); held = 0
    @inbounds for i in eachindex(x)
        open = abs(x[i]) ≥ th
        if open
            held = nh
        elseif held > 0
            held -= 1
            open = true
        end
        target = open ? one(F) : zero(F)
        α = target > g ? αa : αr
        g = α * g + (1 - α) * target
        y[i] = x[i] * g
    end
    return y
end

"""
    SpectralGate(spec; threshold=-60, freqrange=(0, sr÷2), attack=1, release=3) -> DerivedSpec

Noise gate acting on a frequency range of a spectrogram: every bin inside
`freqrange` whose level (dB relative to the spectrogram maximum) is below
`threshold` is attenuated to zero, with the gain of each bin smoothed over
frames by `attack`/`release` frame constants. Bins outside the range are
left untouched. This is the "frequency-range noise gate" idea: a gate that
acts only where the noise lives (a mains hum band, a hiss band) instead of
on the whole signal.
"""
function SpectralGate(s::AbstractSpectrogram; threshold::Real=-60, freqrange::FreqRange=(0, get_sr(s) ÷ 2),
                      attack::Int=1, release::Int=3)
    T = eltype(s)
    S = get_spec(s)
    L = power_to_db(get_spectrum(s) === power ? S : S .^ 2; ref=maximum, amin=1e-10, top_db=nothing)
    idx = _freq_indices(s, freqrange)
    nb, nf = size(S)
    G = ones(T, nb, nf)
    αa = attack  > 1 ? T(exp(-log(9) / attack))  : zero(T)
    αr = release > 1 ? T(exp(-log(9) / release)) : zero(T)
    th = T(threshold)
    @inbounds for k in idx
        g = zero(T)
        for j in 1:nf
            target = L[k, j] ≥ th ? one(T) : zero(T)
            α = target > g ? αa : αr
            g = α * g + (1 - α) * target
            G[k, j] = g
        end
    end
    return _derived(s, S .* G, :gated)
end

# ---------------------------------------------------------------------------- #
#                                     pcen                                     #
# ---------------------------------------------------------------------------- #
"""
    pcen(spec; gain=0.98, bias=2, power=0.5, time_constant=0.4, eps=1e-6, b=nothing) -> DerivedSpec

Per-channel energy normalisation (Wang et al. 2017; librosa `pcen`):
`(S / (eps + M)^gain + bias)^power - bias^power` where `M` is a first-order
IIR smoothing of `S` along frames with coefficient `b`
(default from `time_constant` seconds and the frame hop).
"""
function pcen(s::AbstractSpectrogram; gain::Real=0.98, bias::Real=2, power::Real=0.5,
              time_constant::Real=0.4, eps::Real=1e-6, b::Maybe{Real}=nothing)
    T = eltype(s)
    S = get_spec(s)
    nb, nf = size(S)
    if isnothing(b)
        t = time_constant * get_sr(s) / get_step(s)
        b = (sqrt(1 + 4t^2) - 1) / (2t^2)
    end
    bT, g, bs, p, ε = T(b), T(gain), T(bias), T(power), T(eps)
    M = similar(S)
    @inbounds for k in 1:nb
        m = S[k, 1]
        for j in 1:nf
            m = (1 - bT) * m + bT * S[k, j]
            M[k, j] = m
        end
    end
    P = similar(S)
    @inbounds for i in eachindex(S)
        P[i] = (S[i] / (ε + M[i])^g + bs)^p - bs^p
    end
    return _derived(s, P, :pcen)
end
