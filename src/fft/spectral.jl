# ---------------------------------------------------------------------------- #
#                                    setups                                    #
# ---------------------------------------------------------------------------- #
# The audioFlux keywords of SpectralFlux, SpectralEntropy and
# SpectralBandwidth follow its flux_spectral.c (MIT licence, Copyright (c)
# 2023 libAudioFlux).
struct SpectralSetup <: AbstractSetup
    sr::Int64
end

struct SpectralFluxSetup <: AbstractSetup
    sr::Int64
    p::Float64
    step::Int64
    positive::Bool
    root::Bool
    mean::Bool
end

struct SpectralEntropySetup <: AbstractSetup
    sr::Int64
    normalize::Bool
end

struct SpectralRolloffSetup <: AbstractSetup
    sr::Int64
    threshold::Float64
end

struct SpectralBandwidthSetup <: AbstractSetup
    sr::Int64
    p::Float64
    normalize::Bool
end

# ---------------------------------------------------------------------------- #
#                            per-frame computations                            #
# ---------------------------------------------------------------------------- #
# every kernel takes the spectrogram as (bins × frames) and returns one value
# per frame; all arithmetic stays in T

function _centroid(S::AbstractMatrix{T}, f::AbstractVector) where T
    nb, nf = size(S)
    c = Vector{T}(undef, nf)
    @inbounds for j in 1:nf
        num = zero(T); den = zero(T)
        @simd for k in 1:nb
            num += S[k, j] * T(f[k]); den += S[k, j]
        end
        c[j] = num / den
    end
    return c
end

# Σ (f - c)^q S / Σ S
function _moment(S::AbstractMatrix{T}, f::AbstractVector, c::AbstractVector{T}, q::Int) where T
    nb, nf = size(S)
    m = Vector{T}(undef, nf)
    @inbounds for j in 1:nf
        num = zero(T); den = zero(T)
        @simd for k in 1:nb
            num += (T(f[k]) - c[j])^q * S[k, j]; den += S[k, j]
        end
        m[j] = num / den
    end
    return m
end

_spread(S, f, c) = sqrt.(_moment(S, f, c, 2))

function _skewness(S::AbstractMatrix{T}, f) where T
    c = _centroid(S, f)
    return _moment(S, f, c, 3) ./ _spread(S, f, c) .^ 3
end

function _kurtosis(S::AbstractMatrix{T}, f) where T
    c = _centroid(S, f)
    return _moment(S, f, c, 4) ./ _spread(S, f, c) .^ 4
end

function _crest(S::AbstractMatrix{T}) where T
    nb, nf = size(S)
    [maximum(view(S, :, j)) / (sum(view(S, :, j)) / nb) for j in 1:nf]
end

function _decrease(S::AbstractMatrix{T}) where T
    nb, nf = size(S)
    d = Vector{T}(undef, nf)
    @inbounds for j in 1:nf
        num = zero(T); den = zero(T)
        s1 = S[1, j]
        for k in 2:nb
            num += (S[k, j] - s1) / T(k - 1); den += S[k, j]
        end
        d[j] = num / den
    end
    return d
end

function _entropy(S::AbstractMatrix{T}; normalize::Bool=true) where T
    nb, nf = size(S)
    e = Vector{T}(undef, nf)
    lk = normalize ? log2(T(nb)) : one(T)
    @inbounds for j in 1:nf
        tot = sum(view(S, :, j))
        acc = zero(T)
        if tot > 0
            for k in 1:nb
                x = S[k, j] / tot
                x > 0 && (acc -= x * log2(x))
            end
        end
        e[j] = acc / lk
    end
    return e
end

function _flatness(S::AbstractMatrix{T}) where T
    nb, nf = size(S)
    fl = Vector{T}(undef, nf)
    ε = eps(T)
    @inbounds for j in 1:nf
        lg = zero(T); am = zero(T)
        @simd for k in 1:nb
            lg += log(S[k, j] + ε); am += S[k, j]
        end
        fl[j] = exp(lg / nb) / (am / nb)
    end
    return fl
end

# (Σ d^p)^(1/p) with d = |S_t - S_{t-step}| or its positive part; `mean`
# divides the sum by the number of bins, `root=false` drops the 1/p power
function _flux(S::AbstractMatrix{T}, p::Real; step::Int=1, positive::Bool=false,
               root::Bool=true, mean::Bool=false) where T
    nb, nf = size(S)
    fx = zeros(T, nf)
    pT = T(p)
    @inbounds for j in step+1:nf
        acc = zero(T)
        if p == 2
            @simd for k in 1:nb
                d = S[k, j] - S[k, j-step]
                positive && (d = max(d, zero(T)))
                acc += d^2
            end
        elseif p == 1
            @simd for k in 1:nb
                d = S[k, j] - S[k, j-step]
                acc += positive ? max(d, zero(T)) : abs(d)
            end
        else
            @simd for k in 1:nb
                d = S[k, j] - S[k, j-step]
                acc += (positive ? max(d, zero(T)) : abs(d))^pT
            end
        end
        mean && (acc /= nb)
        fx[j] = !root || p == 1 ? acc : (p == 2 ? sqrt(acc) : acc^(one(T) / pT))
    end
    return fx
end

function _rolloff(S::AbstractMatrix{T}, f::AbstractVector, threshold::Real) where T
    nb, nf = size(S)
    r = Vector{T}(undef, nf)
    th = T(threshold)
    @inbounds for j in 1:nf
        target = sum(view(S, :, j)) * th
        acc = zero(T); k = 1
        while k < nb
            acc += S[k, j]
            acc >= target && break
            k += 1
        end
        r[j] = T(f[k])
    end
    return r
end

function _slope(S::AbstractMatrix{T}, f::AbstractVector) where T
    nb, nf = size(S)
    fm = T(sum(f) / nb)
    df = T[T(x) - fm for x in f]
    den = sum(abs2, df)
    s = Vector{T}(undef, nf)
    @inbounds for j in 1:nf
        sm = sum(view(S, :, j)) / nb
        acc = zero(T)
        @simd for k in 1:nb
            acc += (S[k, j] - sm) * df[k]
        end
        s[j] = acc / den
    end
    return s
end

function _bandwidth(S::AbstractMatrix{T}, f::AbstractVector, p::Real; normalize::Bool=true) where T
    c = _centroid(S, f)
    nb, nf = size(S)
    b = Vector{T}(undef, nf)
    @inbounds for j in 1:nf
        tot = normalize ? sum(view(S, :, j)) : one(T)
        acc = zero(T)
        for k in 1:nb
            acc += S[k, j] / tot * abs(T(f[k]) - c[j])^p
        end
        b[j] = acc^(one(T) / p)
    end
    return b
end

# ---------------------------------------------------------------------------- #
#                                  descriptors                                 #
# ---------------------------------------------------------------------------- #
# every descriptor is a struct with one value per frame, its parent stage and
# an info record; the macro only writes the boilerplate.
macro descriptor(name, setup, doc)
    quote
        Core.@doc $doc struct $(esc(name)){F,T<:AudioData} <: AbstractSpectral
            spec   :: Vector{T}
            parent :: F
            info   :: $(esc(setup))
        end
        Base.eltype(::$(esc(name)){F,T}) where {F,T} = T
        Audio911.get_parent(x::$(esc(name))) = x.parent
        function Base.show(io::IO, x::$(esc(name)){F,T}) where {F,T}
            print(io, $(string(name)), "{", nameof(F), ",", T, "}(", length(x.spec), " frames)")
        end
    end
end

@descriptor SpectralCentroid SpectralSetup """
    SpectralCentroid(spec::AbstractSpectrogram) -> SpectralCentroid

Spectral centroid `Σ f·S / Σ S` of every frame, in Hz (MATLAB `spectralCentroid`).
"""
SpectralCentroid(s::AbstractSpectrogram) =
    SpectralCentroid{typeof(s),eltype(s)}(_centroid(get_spec(s), get_freq(s)), s, SpectralSetup(get_sr(s)))

@descriptor SpectralCrest SpectralSetup """
    SpectralCrest(spec::AbstractSpectrogram) -> SpectralCrest

Spectral crest `max(S) / mean(S)` of every frame (MATLAB `spectralCrest`).
"""
SpectralCrest(s::AbstractSpectrogram) =
    SpectralCrest{typeof(s),eltype(s)}(_crest(get_spec(s)), s, SpectralSetup(get_sr(s)))

@descriptor SpectralDecrease SpectralSetup """
    SpectralDecrease(spec::AbstractSpectrogram) -> SpectralDecrease

Spectral decrease `Σ_{k≥2} (S_k - S_1)/(k-1) / Σ_{k≥2} S_k` (MATLAB `spectralDecrease`).
"""
SpectralDecrease(s::AbstractSpectrogram) =
    SpectralDecrease{typeof(s),eltype(s)}(_decrease(get_spec(s)), s, SpectralSetup(get_sr(s)))

@descriptor SpectralEntropy SpectralEntropySetup """
    SpectralEntropy(spec::AbstractSpectrogram; normalize=true) -> SpectralEntropy

Normalised spectral entropy `-Σ p log2 p / log2(nbins)` with `p = S/ΣS`
(MATLAB `spectralEntropy`). `normalize=false` drops the division by
`log2(nbins)` (audioFlux `entropy(is_norm=False)`). Empty frames give 0.
"""
SpectralEntropy(s::AbstractSpectrogram; normalize::Bool=true) =
    SpectralEntropy{typeof(s),eltype(s)}(_entropy(get_spec(s); normalize), s, SpectralEntropySetup(get_sr(s), normalize))

@descriptor SpectralFlatness SpectralSetup """
    SpectralFlatness(spec::AbstractSpectrogram) -> SpectralFlatness

Spectral flatness, geometric over arithmetic mean of the bins (MATLAB `spectralFlatness`).
"""
SpectralFlatness(s::AbstractSpectrogram) =
    SpectralFlatness{typeof(s),eltype(s)}(_flatness(get_spec(s)), s, SpectralSetup(get_sr(s)))

@descriptor SpectralFlux SpectralFluxSetup """
    SpectralFlux(spec::AbstractSpectrogram; p=2, step=1, positive=false, root=true, mean=false) -> SpectralFlux

Spectral flux, the `p`-norm of the difference between frame `t` and frame
`t - step` (MATLAB `spectralFlux` with `NormType=p`); the first `step`
frames are 0. The other keywords give audioFlux's variants
(`flux(step, p, is_positive, is_exp, tp)`): `positive=true` keeps only the
increases (half-wave rectification), `root=false` returns `Σ d^p` without
the `1/p` power (audioFlux's default), `mean=true` divides the sum by the
number of bins.
"""
SpectralFlux(s::AbstractSpectrogram; p::Real=2, step::Int=1, positive::Bool=false,
             root::Bool=true, mean::Bool=false) = begin
    step ≥ 1 || throw(ArgumentError("step must be ≥ 1"))
    p > 0 || throw(ArgumentError("p must be positive"))
    SpectralFlux{typeof(s),eltype(s)}(_flux(get_spec(s), p; step, positive, root, mean), s,
                                      SpectralFluxSetup(get_sr(s), Float64(p), step, positive, root, mean))
end

@descriptor SpectralKurtosis SpectralSetup """
    SpectralKurtosis(spec::AbstractSpectrogram) -> SpectralKurtosis

Spectral kurtosis, fourth standardised moment of the spectrum around its
centroid (MATLAB `spectralKurtosis`).
"""
SpectralKurtosis(s::AbstractSpectrogram) =
    SpectralKurtosis{typeof(s),eltype(s)}(_kurtosis(get_spec(s), get_freq(s)), s, SpectralSetup(get_sr(s)))

@descriptor SpectralRolloff SpectralRolloffSetup """
    SpectralRolloff(spec::AbstractSpectrogram; threshold=0.95) -> SpectralRolloff

Roll-off frequency: the bin below which `threshold` of the frame energy lies
(MATLAB `spectralRolloffPoint`), in Hz.
"""
SpectralRolloff(s::AbstractSpectrogram; threshold::Real=0.95) =
    SpectralRolloff{typeof(s),eltype(s)}(_rolloff(get_spec(s), get_freq(s), threshold), s,
                                         SpectralRolloffSetup(get_sr(s), Float64(threshold)))

@descriptor SpectralSkewness SpectralSetup """
    SpectralSkewness(spec::AbstractSpectrogram) -> SpectralSkewness

Spectral skewness, third standardised moment around the centroid (MATLAB `spectralSkewness`).
"""
SpectralSkewness(s::AbstractSpectrogram) =
    SpectralSkewness{typeof(s),eltype(s)}(_skewness(get_spec(s), get_freq(s)), s, SpectralSetup(get_sr(s)))

@descriptor SpectralSlope SpectralSetup """
    SpectralSlope(spec::AbstractSpectrogram) -> SpectralSlope

Spectral slope, the linear regression slope of the spectrum against
frequency (MATLAB `spectralSlope`).
"""
SpectralSlope(s::AbstractSpectrogram) =
    SpectralSlope{typeof(s),eltype(s)}(_slope(get_spec(s), get_freq(s)), s, SpectralSetup(get_sr(s)))

@descriptor SpectralSpread SpectralSetup """
    SpectralSpread(spec::AbstractSpectrogram) -> SpectralSpread

Spectral spread, standard deviation of the spectrum around its centroid in
Hz (MATLAB `spectralSpread`).
"""
function SpectralSpread(s::AbstractSpectrogram)
    S, f = get_spec(s), get_freq(s)
    SpectralSpread{typeof(s),eltype(s)}(_spread(S, f, _centroid(S, f)), s, SpectralSetup(get_sr(s)))
end

@descriptor SpectralBandwidth SpectralBandwidthSetup """
    SpectralBandwidth(spec::AbstractSpectrogram; p=2, normalize=true) -> SpectralBandwidth

librosa's `spectral_bandwidth`: `(Σ p̃(k) |f(k) - centroid|^p)^(1/p)` with the
spectrum normalised to unit sum. For `p=2` it equals [`SpectralSpread`](@ref).
`normalize=false` weights by the spectrum itself (audioFlux `band_width`).
"""
SpectralBandwidth(s::AbstractSpectrogram; p::Real=2, normalize::Bool=true) =
    SpectralBandwidth{typeof(s),eltype(s)}(_bandwidth(get_spec(s), get_freq(s), p; normalize), s,
                                           SpectralBandwidthSetup(get_sr(s), Float64(p), normalize))

# ---------------------------------------------------------------------------- #
#                                    methods                                   #
# ---------------------------------------------------------------------------- #
"""
    get_data(s::AbstractSpectral) -> Vector

One value per frame.
"""
@inline get_data(s::AbstractSpectral)  = s.spec
@inline get_setup(s::AbstractSpectral) = s.info
@inline get_sr(s::AbstractSpectral)    = s.info.sr
@inline get_nframes(s::AbstractSpectral) = length(s.spec)
