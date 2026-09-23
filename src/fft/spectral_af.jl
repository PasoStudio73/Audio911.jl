# ---------------------------------------------------------------------------- #
#                   audioFlux spectral descriptors and novelties               #
# ---------------------------------------------------------------------------- #
# Ports of src/flux_spectral.c and src/feature/spectral_algorithm.c of
# audioFlux (MIT licence, Copyright (c) 2023 libAudioFlux). Every descriptor
# reads the spectrogram as given (power or magnitude) unless its docstring
# says otherwise, and returns one value per frame; frames that have no
# predecessor for a difference are 0.

struct SpectralParamSetup <: AbstractSetup
    sr::Int64
    params::NamedTuple
end

# magnitude and power views of a spectrogram
_mag(s::AbstractSpectrogram) = get_spectrum(s) === power ? sqrt.(get_spec(s)) : get_spec(s)
_pow(s::AbstractSpectrogram) = get_spectrum(s) === power ? get_spec(s) : get_spec(s) .^ 2

_descr(D, s, v, params) = D{typeof(s),eltype(s)}(v, s, SpectralParamSetup(get_sr(s), params))

# ------------------------------------------------------------------ energy ---
function _energy(P::AbstractMatrix{T}, log::Bool, γ::Real) where T
    nb, nf = size(P)
    g = T(γ)
    return T[sum(k -> log ? Base.log(1 + g * P[k, j]) : P[k, j], 1:nb) / nb for j in 1:nf]
end

@descriptor SpectralEnergy SpectralParamSetup """
    SpectralEnergy(spec; log=false, gamma=10) -> SpectralEnergy

Mean power over the bins, `mean(|X|²)`, or `mean(log(1 + γ|X|²))` with
`log=true` (audioFlux `energy`).
"""
SpectralEnergy(s::AbstractSpectrogram; log::Bool=false, gamma::Real=10) =
    _descr(SpectralEnergy, s, _energy(_pow(s), log, gamma), (; log, gamma=Float64(gamma)))

# --------------------------------------------------------------------- rms ---
@descriptor SpectralRms SpectralParamSetup """
    SpectralRms(spec) -> SpectralRms

Root-mean-square level from the magnitude spectrum, audioFlux's `rms`:
`sqrt(2 Σ w_k |X_k|² / n²)` over the `n` bins, with the DC bin (and the
last bin when `n` is even) at half weight. [`Rms`](@ref) is librosa's
normalisation by the FFT length.
"""
function SpectralRms(s::AbstractSpectrogram)
    P = _pow(s); T = eltype(s)
    nb, nf = size(P)
    v = Vector{T}(undef, nf)
    @inbounds for j in 1:nf
        acc = zero(T)
        for k in 1:nb
            w = (k == 1 || (iseven(nb) && k == nb)) ? T(0.5) : one(T)
            acc += w * P[k, j]
        end
        v[j] = sqrt(2acc / nb^2)
    end
    return _descr(SpectralRms, s, v, (;))
end

# --------------------------------------------------------------------- hfc ---
@descriptor SpectralHfc SpectralParamSetup """
    SpectralHfc(spec) -> SpectralHfc

High-frequency content `Σ k S_k` with the bin index `k` counted from 0
(Masri 1996; audioFlux `hfc`).
"""
function SpectralHfc(s::AbstractSpectrogram)
    S = get_spec(s); T = eltype(s)
    v = T[sum(k -> (k - 1) * S[k, j], axes(S, 1)) for j in axes(S, 2)]
    return _descr(SpectralHfc, s, v, (;))
end

# ------------------------------------------------------------------ sd, sf ---
function _diffsum(S::AbstractMatrix{T}, step::Int, positive::Bool, sq::Bool) where T
    nb, nf = size(S)
    v = zeros(T, nf)
    @inbounds for j in step+1:nf
        acc = zero(T)
        for k in 1:nb
            d = S[k, j] - S[k, j-step]
            d = positive ? max(d, zero(T)) : abs(d)
            acc += sq ? d^2 : d
        end
        v[j] = acc
    end
    return v
end

@descriptor SpectralSd SpectralParamSetup """
    SpectralSd(spec; step=1, positive=false) -> SpectralSd

Spectral difference `Σ |S_t - S_{t-step}|` (`positive=true`: only the
increases), audioFlux `sd`.
"""
SpectralSd(s::AbstractSpectrogram; step::Int=1, positive::Bool=false) =
    _descr(SpectralSd, s, _diffsum(get_spec(s), step, positive, false), (; step, positive))

@descriptor SpectralSf SpectralParamSetup """
    SpectralSf(spec; step=1, positive=false) -> SpectralSf

Squared spectral difference `Σ (S_t - S_{t-step})²` (audioFlux `sf`).
"""
SpectralSf(s::AbstractSpectrogram; step::Int=1, positive::Bool=false) =
    _descr(SpectralSf, s, _diffsum(get_spec(s), step, positive, true), (; step, positive))

# --------------------------------------------------------------------- mkl ---
@descriptor SpectralMkl SpectralParamSetup """
    SpectralMkl(spec; mean=false) -> SpectralMkl

Modified Kullback-Leibler divergence between consecutive frames,
`Σ log(1 + S_t / (S_{t-1} + 10⁻¹⁶))` (Hainsworth & Macleod 2003; audioFlux
`mkl`), summed or (`mean=true`) averaged over the bins.
"""
function SpectralMkl(s::AbstractSpectrogram; mean::Bool=false)
    S = get_spec(s); T = eltype(s)
    nb, nf = size(S)
    v = zeros(T, nf)
    @inbounds for j in 2:nf
        acc = zero(T)
        for k in 1:nb
            acc += log(1 + S[k, j] / (S[k, j-1] + T(1e-16)))
        end
        v[j] = mean ? acc / nb : acc
    end
    return _descr(SpectralMkl, s, v, (; mean))
end

# ------------------------------------------------ phase and complex deviation ---
# magnitude and (unwrapped over nothing: raw) phase of a front end's complex coefficients
function _magphase(s::AbstractSpectrogram)
    C = get_complex(s)
    return abs.(C), angle.(C)
end

function _phase_dev(M::AbstractMatrix{T}, Φ::AbstractMatrix{T}, weight::Bool, norm::Bool) where T
    nb, nf = size(M)
    v = zeros(T, nf)
    @inbounds for j in 3:nf
        acc = zero(T)
        for k in 1:nb
            d = abs(Φ[k, j] - 2Φ[k, j-1] + Φ[k, j-2])
            (weight || norm) && (d *= M[k, j])
            acc += d
        end
        acc /= nb
        if norm
            acc /= sum(view(M, :, j)) / nb + T(1e-16)
        end
        v[j] = acc
    end
    return v
end

@descriptor SpectralPd SpectralParamSetup """
    SpectralPd(spec) -> SpectralPd

Phase deviation (Bello et al. 2004; audioFlux `pd`): the mean over the bins
of `|φ_t - 2φ_{t-1} + φ_{t-2}|`, the second difference of the phase of the
complex coefficients ([`get_complex`](@ref), so `spec` must be a front end
that provides them: `Stft`, `Cqt`, `Cwt`, ...). The first two frames are 0.
"""
SpectralPd(s::AbstractSpectrogram) = (MΦ = _magphase(s); _descr(SpectralPd, s, _phase_dev(MΦ..., false, false), (;)))

@descriptor SpectralWpd SpectralParamSetup """
    SpectralWpd(spec) -> SpectralWpd

Weighted phase deviation (Dixon 2006; audioFlux `wpd`): the phase deviation
of every bin weighted by its magnitude, averaged over the bins.
"""
SpectralWpd(s::AbstractSpectrogram) = (MΦ = _magphase(s); _descr(SpectralWpd, s, _phase_dev(MΦ..., true, false), (;)))

@descriptor SpectralNwpd SpectralParamSetup """
    SpectralNwpd(spec) -> SpectralNwpd

Normalised weighted phase deviation (audioFlux `nwpd`): [`SpectralWpd`](@ref)
divided by the mean magnitude of the frame.
"""
SpectralNwpd(s::AbstractSpectrogram) = (MΦ = _magphase(s); _descr(SpectralNwpd, s, _phase_dev(MΦ..., false, true), (;)))

function _complex_dev(M::AbstractMatrix{T}, Φ::AbstractMatrix{T}, rectify::Bool) where T
    nb, nf = size(M)
    v = zeros(T, nf)
    @inbounds for j in 2:nf
        acc = zero(T)
        for k in 1:nb
            rectify && M[k, j] ≤ M[k, j-1] && continue
            z = M[k, j] * cis(Φ[k, j])
            j > 2 && (z -= M[k, j-1] * cis(2Φ[k, j-1] - Φ[k, j-2]))
            acc += abs(z)
        end
        v[j] = acc
    end
    return v
end

@descriptor SpectralCd SpectralParamSetup """
    SpectralCd(spec) -> SpectralCd

Complex-domain deviation (Duxbury et al. 2003; audioFlux `cd`): the sum
over the bins of `|X_t - X̂_t|`, where `X̂_t` continues the previous frame
with the phase extrapolated linearly, `|X_{t-1}| e^{i(2φ_{t-1} - φ_{t-2})}`.
Needs the complex coefficients, like [`SpectralPd`](@ref).
"""
SpectralCd(s::AbstractSpectrogram) = (MΦ = _magphase(s); _descr(SpectralCd, s, _complex_dev(MΦ..., false), (;)))

@descriptor SpectralRcd SpectralParamSetup """
    SpectralRcd(spec) -> SpectralRcd

Rectified complex-domain deviation (Dixon 2006; audioFlux `rcd`): the
complex deviation over the bins whose magnitude increased.
"""
SpectralRcd(s::AbstractSpectrogram) = (MΦ = _magphase(s); _descr(SpectralRcd, s, _complex_dev(MΦ..., true), (;)))

# --------------------------------------------------------------- broadband ---
@descriptor SpectralBroadband SpectralParamSetup """
    SpectralBroadband(spec; threshold=0) -> SpectralBroadband

Number of bins whose level rose by more than `threshold` dB,
`10 log10(S_t / S_{t-1}) > threshold` (Duxbury et al. 2002; audioFlux
`broadband`).
"""
function SpectralBroadband(s::AbstractSpectrogram; threshold::Real=0)
    S = get_spec(s); T = eltype(s)
    nb, nf = size(S)
    v = zeros(T, nf)
    th = T(threshold)
    @inbounds for j in 2:nf, k in 1:nb
        10 * log10(S[k, j] / S[k, j-1]) > th && (v[j] += 1)
    end
    return _descr(SpectralBroadband, s, v, (; threshold=Float64(threshold)))
end

# ----------------------------------------------------------------- novelty ---
@descriptor SpectralNovelty SpectralParamSetup """
    SpectralNovelty(spec; step=1, threshold=0, method=:sub, data=:value) -> SpectralNovelty

Spectral novelty against frame `t - step` (audioFlux `novelty`): for every
bin the difference `S_t - S_{t-step}` (`method=:sub`), `log(S_t/S_{t-step})`
(`:entropy`), `S_t log(S_t/S_{t-step})` (`:kl`) or the Itakura-Saito term
`S_t/S_{t-step} - log(S_t/S_{t-step}) - 1` (`:is`); the terms above
`threshold` are summed (`data=:value`) or counted (`data=:number`).
"""
function SpectralNovelty(s::AbstractSpectrogram; step::Int=1, threshold::Real=0,
                         method::Symbol=:sub, data::Symbol=:value)
    method in (:sub, :entropy, :kl, :is) || throw(ArgumentError("method must be :sub, :entropy, :kl or :is, got :$method"))
    data in (:value, :number) || throw(ArgumentError("data must be :value or :number, got :$data"))
    step ≥ 1 || throw(ArgumentError("step must be ≥ 1"))
    S = get_spec(s); T = eltype(s)
    nb, nf = size(S)
    v = zeros(T, nf)
    th = T(threshold); ε = T(1e-16)
    @inbounds for j in step+1:nf
        acc = zero(T)
        for k in 1:nb
            c, p = S[k, j], S[k, j-step]
            d = method === :sub ? c - p :
                method === :entropy ? log(c / (p + ε)) :
                method === :kl ? c * log(c / (p + ε)) :
                c / (p + ε) - log(c / (p + ε)) - 1
            d > th && (acc += data === :value ? d : one(T))
        end
        v[j] = acc
    end
    return _descr(SpectralNovelty, s, v, (; step, threshold=Float64(threshold), method, data))
end

# ---------------------------------------------------------------- eef, eer ---
@descriptor SpectralEef SpectralParamSetup """
    SpectralEef(spec; normalize=false) -> SpectralEef

Energy-entropy feature `sqrt(1 + |E · H|)` with `E` the
[`SpectralEnergy`](@ref) and `H` the [`SpectralEntropy`](@ref) (normalised
or not), audioFlux `eef`.
"""
function SpectralEef(s::AbstractSpectrogram; normalize::Bool=false)
    E = _energy(_pow(s), false, 0); H = _entropy(get_spec(s); normalize)
    return _descr(SpectralEef, s, sqrt.(1 .+ abs.(E .* H)), (; normalize))
end

@descriptor SpectralEer SpectralParamSetup """
    SpectralEer(spec; normalize=false, gamma=1) -> SpectralEer

Energy-entropy ratio `sqrt(1 + |log(1 + γE) / H|)` (audioFlux `eer`).
"""
function SpectralEer(s::AbstractSpectrogram; normalize::Bool=false, gamma::Real=1)
    T = eltype(s)
    E = _energy(_pow(s), false, 0); H = _entropy(get_spec(s); normalize)
    return _descr(SpectralEer, s, sqrt.(1 .+ abs.(log.(1 .+ T(gamma) .* E) ./ H)), (; normalize, gamma=Float64(gamma)))
end

# -------------------------------------------------------------- statistics ---
@descriptor SpectralMax SpectralParamSetup """
    SpectralMax(spec) -> SpectralMax

Largest bin value of every frame (audioFlux `max`, value output);
[`SpectralPeak`](@ref) gives its frequency.
"""
SpectralMax(s::AbstractSpectrogram) =
    _descr(SpectralMax, s, vec(maximum(get_spec(s), dims=1)), (;))

@descriptor SpectralPeak SpectralParamSetup """
    SpectralPeak(spec) -> SpectralPeak

Frequency (Hz) of the largest bin of every frame (audioFlux `max`,
frequency output).
"""
function SpectralPeak(s::AbstractSpectrogram)
    S = get_spec(s); f = get_freq(s); T = eltype(s)
    return _descr(SpectralPeak, s, T[f[argmax(view(S, :, j))] for j in axes(S, 2)], (;))
end

@descriptor SpectralMean SpectralParamSetup """
    SpectralMean(spec) -> SpectralMean

Mean bin value of every frame (audioFlux `mean`, value output; its
frequency output is the mean of the band frequencies, the same for every
frame).
"""
SpectralMean(s::AbstractSpectrogram) =
    _descr(SpectralMean, s, vec(sum(get_spec(s), dims=1)) ./ get_nbins(s), (;))

@descriptor SpectralVar SpectralParamSetup """
    SpectralVar(spec) -> SpectralVar

Sample variance of the bin values of every frame (audioFlux `var`, value
output).
"""
function SpectralVar(s::AbstractSpectrogram)
    S = get_spec(s); T = eltype(s)
    nb = size(S, 1)
    v = T[(m = sum(view(S, :, j)) / nb; sum(k -> (S[k, j] - m)^2, 1:nb) / (nb - 1)) for j in axes(S, 2)]
    return _descr(SpectralVar, s, v, (;))
end
