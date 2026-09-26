# ---------------------------------------------------------------------------- #
#                                onset strength                                #
# ---------------------------------------------------------------------------- #
# Novelty and the peak-picking windows follow audioFlux's mir/onset_algorithm.c
# (MIT licence, Copyright (c) 2023 libAudioFlux).
struct OnsetSetup <: AbstractSetup
    sr::Int64
    lag::Int64
    max_size::Int64
    detrend::Bool
end

# running maximum over `size` bins along the frequency axis (centered)
function _max_filter_freq(S::AbstractMatrix{T}, size::Int) where T
    size ≤ 1 && return S
    nb, nf = Base.size(S)
    h = size ÷ 2
    R = similar(S)
    @inbounds for j in 1:nf, k in 1:nb
        lo, hi = max(1, k - h), min(nb, k + (size - 1 - h))
        R[k, j] = maximum(view(S, lo:hi, j))
    end
    return R
end

@descriptor OnsetStrength OnsetSetup """
    OnsetStrength(spec; lag=1, max_size=1, detrend=false, aggregate=mean) -> OnsetStrength

Spectral-flux onset strength envelope (Böck & Widmer 2013; librosa
`onset_strength`): the spectrogram is converted to dB (`power_to_db` with
`top_db=80`), the positive difference between frame `t` and frame `t - lag`
(against a running frequency maximum of width `max_size`) is aggregated
over the bins with `aggregate`, and the first `lag` frames are zero.
`detrend=true` removes the DC trend with the `[1, -1] / [1, -0.99]` filter.
librosa computes this on a mel spectrogram, so pass a `MelSpec` to match it;
the window-centering shift librosa applies is not reproduced.
"""
function OnsetStrength(s::AbstractSpectrogram; lag::Int=1, max_size::Int=1, detrend::Bool=false,
                       aggregate::Function=mean)
    T = eltype(s)
    lag ≥ 1 || throw(ArgumentError("lag must be ≥ 1"))
    max_size ≥ 1 || throw(ArgumentError("max_size must be ≥ 1"))
    S = get_spectrum(s) === power ? get_spec(s) : get_spec(s) .^ 2
    D = power_to_db(S; ref=1, amin=1e-10, top_db=80)
    R = _max_filter_freq(D, max_size)
    nb, nf = size(D)
    env = zeros(T, nf)
    @inbounds for j in lag+1:nf
        env[j] = T(aggregate(max.(view(D, :, j) .- view(R, :, j - lag), zero(T))))
    end
    if detrend
        y = zero(T); xp = zero(T)
        @inbounds for j in 1:nf
            y = env[j] - xp + T(0.99) * y
            xp = env[j]
            env[j] = y
        end
    end
    return OnsetStrength{typeof(s),T}(env, s, OnsetSetup(get_sr(s), lag, max_size, detrend))
end

# ---------------------------------------------------------------------------- #
#                                  peak picking                                #
# ---------------------------------------------------------------------------- #
"""
    peak_pick(x; pre_max, post_max, pre_avg, post_avg, delta, wait) -> Vector{Int}

Indices `n` such that `x[n]` is the maximum of `x[n-pre_max : n+post_max-1]`,
`x[n] ≥ mean(x[n-pre_avg : n+post_avg-1]) + delta`, and more than `wait`
samples after the previous pick (librosa `peak_pick`, whose windows end
before `n + post_max`; audioFlux's onset peak picking is the same rule).
"""
function peak_pick(x::AbstractVector{T}; pre_max::Int, post_max::Int, pre_avg::Int, post_avg::Int,
                   delta::Real, wait::Int) where {T<:Real}
    n = length(x)
    peaks = Int[]
    last = -wait - 1
    @inbounds for i in 1:n
        lo, hi = max(1, i - pre_max), min(n, i + post_max - 1)
        x[i] == maximum(view(x, lo:max(hi, i))) || continue
        alo, ahi = max(1, i - pre_avg), min(n, i + post_avg - 1)
        x[i] >= mean(view(x, alo:max(ahi, i))) + delta || continue
        i - last > wait || continue
        push!(peaks, i)
        last = i
    end
    return peaks
end

"""
    onset_detect(env; delta=0.07, normalize=true, kwargs...) -> Vector{Int}

Frame indices of the onsets picked from an onset envelope (an
[`OnsetStrength`](@ref), a [`Novelty`](@ref) or any one-value-per-frame
descriptor) with [`peak_pick`](@ref), with librosa's and audioFlux's
defaults: `pre_max` 30 ms, `post_max` 1 frame, `pre_avg` 100 ms,
`post_avg` 100 ms + 1 frame, `wait` 30 ms, each rounded down to frames. The
envelope is normalised to its maximum first when `normalize=true`. Use
`get_times(env)[idx]` for the onset times in seconds.
"""
function onset_detect(env::AbstractSpectral; delta::Real=0.07, normalize::Bool=true,
                      pre_max::Maybe{Int}=nothing, post_max::Maybe{Int}=nothing,
                      pre_avg::Maybe{Int}=nothing, post_avg::Maybe{Int}=nothing, wait::Maybe{Int}=nothing)
    fps = get_sr(env) / get_step(env)
    x = get_data(env)
    if normalize
        m = maximum(x)
        m > 0 && (x = x ./ m)
    end
    return peak_pick(x;
        pre_max=something(pre_max, floor(Int, 0.03fps)),
        post_max=something(post_max, floor(Int, 0.0fps) + 1),
        pre_avg=something(pre_avg, floor(Int, 0.10fps)),
        post_avg=something(post_avg, floor(Int, 0.10fps) + 1),
        delta, wait=something(wait, floor(Int, 0.03fps)))
end

# ---------------------------------------------------------------------------- #
#                            novelty onset envelopes                           #
# ---------------------------------------------------------------------------- #
struct NoveltySetup <: AbstractSetup
    sr::Int64
    method::Base.Callable
    filter_order::Int64
end

const _PHASE_NOVELTIES = (SpectralPd, SpectralWpd, SpectralNwpd, SpectralCd, SpectralRcd)

@descriptor Novelty NoveltySetup """
    Novelty(spec; method=SpectralFlux, filter_order=1, kwargs...) -> Novelty

Onset envelope from any spectral novelty, audioFlux's `Onset`: `method` is
one of [`SpectralFlux`](@ref) (audioFlux's `FLUX`), [`SpectralHfc`](@ref),
[`SpectralSd`](@ref), [`SpectralSf`](@ref), [`SpectralMkl`](@ref),
[`SpectralBroadband`](@ref), the phase and complex-domain deviations
[`SpectralPd`](@ref), [`SpectralWpd`](@ref), [`SpectralNwpd`](@ref),
[`SpectralCd`](@ref), [`SpectralRcd`](@ref), or any other descriptor, called
with `kwargs`. With `filter_order > 1` the spectrogram is first replaced by
its running maximum over `filter_order` bins (not for the phase methods).
The envelope is scaled to `[0, 1]` (minus its minimum, over its maximum);
[`onset_detect`](@ref) picks the onsets.

```julia
mel = MelSpec(stft; nbands=128)
env = Novelty(mel; method=SpectralFlux, p=1, positive=true, root=false)   # audioFlux's FLUX
env = Novelty(Stft(audio; keep_complex=true); method=SpectralCd)
idx = onset_detect(env)
```
"""
function Novelty(s::AbstractSpectrogram; method=SpectralFlux, filter_order::Int=1, kwargs...)
    filter_order ≥ 1 || throw(ArgumentError("filter_order must be ≥ 1"))
    T = eltype(s)
    src = s
    if filter_order > 1 && !(method in _PHASE_NOVELTIES)
        src = _derived(s, _max_filter_freq(get_spec(s), filter_order), :maxfiltered)
    end
    v = Vector{T}(get_data(method(src; kwargs...)))
    lo, hi = extrema(v)
    v .-= lo
    hi - lo > 0 && (v ./= hi - lo)
    return Novelty{typeof(s),T}(v, s, NoveltySetup(get_sr(s), method, filter_order))
end

# ---------------------------------------------------------------------------- #
#                                   tempogram                                  #
# ---------------------------------------------------------------------------- #
struct TempogramSetup <: AbstractSetup
    sr::Int64
    win_length::Int64
end

"""
    Tempogram{F,T} <: AbstractSpectrogram

Local auto-correlation tempogram (Grosche, Müller & Kurth 2010; librosa
`tempogram`), `win_length × frames`; row `k` is lag `k - 1` frames.
`get_freq` returns the tempo of every lag in BPM (`Inf` at lag 0).
"""
struct Tempogram{F,T<:AbstractFloat} <: AbstractSpectrogram
    spec   :: Matrix{T}
    parent :: F
    info   :: TempogramSetup
end

Base.eltype(::Tempogram{F,T}) where {F,T} = T
get_data(t::Tempogram)   = t.spec'
get_spec(t::Tempogram)   = t.spec
get_setup(t::Tempogram)  = t.info
get_sr(t::Tempogram)     = t.info.sr
get_parent(t::Tempogram) = t.parent
get_freq(t::Tempogram)   = tempo_frequencies(eltype(t), size(t.spec, 1), get_step(t), get_sr(t))
Base.show(io::IO, t::Tempogram{F,T}) where {F,T} =
    print(io, "Tempogram{$(nameof(F)),$T}($(size(t.spec, 2)) frames × $(size(t.spec, 1)) lags)")

"""
    tempo_frequencies([T,] n, hop, sr) -> Vector

BPM of lags `0:n-1` frames: `60 * sr / (hop * lag)` (librosa `tempo_frequencies`).
"""
tempo_frequencies(::Type{T}, n::Int, hop::Int, sr::Int) where T =
    T[k == 0 ? T(Inf) : T(60 * sr / (hop * k)) for k in 0:n-1]
tempo_frequencies(n::Int, hop::Int, sr::Int) = tempo_frequencies(Float64, n, hop, sr)

"""
    Tempogram(env::OnsetStrength; win_length=384, center=true) -> Tempogram

Windowed (Hann) auto-correlation of the onset envelope around every frame,
each column normalised to its maximum (librosa `tempogram`).
"""
function Tempogram(env::OnsetStrength; win_length::Int=384, center::Bool=true)
    T = eltype(env)
    x = get_data(env)
    n = length(x)
    win_length ≥ 2 || throw(ArgumentError("win_length must be ≥ 2"))
    pad = center ? win_length ÷ 2 : 0
    xp = vcat(zeros(T, pad), x, zeros(T, pad))
    w  = Vector{T}(hanning(win_length))
    nfr = center ? n : max(n - win_length + 1, 0)
    out = Matrix{T}(undef, win_length, nfr)
    m = nextpow(2, 2win_length)
    plan = plan_rfft(zeros(T, m))
    buf = zeros(T, m)
    F = Vector{Complex{T}}(undef, m ÷ 2 + 1)
    @inbounds for j in 1:nfr
        fill!(buf, zero(T))
        for i in 1:win_length
            buf[i] = xp[j + i - 1] * w[i]
        end
        mul!(F, plan, buf)
        r = irfft(abs2.(F), m)
        col = view(r, 1:win_length)
        mx = maximum(abs, col)
        out[:, j] .= mx > 0 ? col ./ mx : col
    end
    return Tempogram{typeof(env),T}(out, env, TempogramSetup(get_sr(env), win_length))
end

# ---------------------------------------------------------------------------- #
#                                     tempo                                    #
# ---------------------------------------------------------------------------- #
"""
    tempo(env::OnsetStrength; start_bpm=120, std_bpm=1, ac_size=8, max_tempo=320, aggregate=mean) -> Float64

Global tempo estimate in BPM (Ellis 2007; librosa `beat.tempo`): the
tempogram over `ac_size` seconds is aggregated over time, weighted by a
log-normal prior centred on `start_bpm` with width `std_bpm` octaves, and
the strongest lag below `max_tempo` is returned.
"""
function tempo(env::OnsetStrength; start_bpm::Real=120, std_bpm::Real=1, ac_size::Real=8,
               max_tempo::Maybe{Real}=320, aggregate::Function=mean)
    hop, sr = get_step(env), get_sr(env)
    win_length = max(2, round(Int, ac_size * sr / hop))
    tg = Tempogram(env; win_length)
    agg = [aggregate(view(get_spec(tg), k, :)) for k in 1:win_length]
    bpms = tempo_frequencies(win_length, hop, sr)
    prior = [k == 1 ? 0.0 : exp(-0.5 * ((log2(bpms[k]) - log2(start_bpm)) / std_bpm)^2) for k in 1:win_length]
    if !isnothing(max_tempo)
        for k in 2:win_length
            bpms[k] > max_tempo && (prior[k] = 0.0)
        end
    end
    return bpms[argmax(agg .* prior)]
end

# ---------------------------------------------------------------------------- #
#                                 beat tracking                                #
# ---------------------------------------------------------------------------- #
function _local_score(env::AbstractVector{T}, period::Int) where T
    σ = std(env)
    x = σ > 0 ? env ./ σ : copy(env)
    w = [exp(-0.5 * (k * 32 / period)^2) for k in -period:period]
    n = length(x); h = period
    out = zeros(T, n)
    @inbounds for i in 1:n, k in -h:h
        j = i + k
        1 ≤ j ≤ n && (out[i] += x[j] * w[k + h + 1])
    end
    return out
end

"""
    beat_track(env::OnsetStrength; bpm=nothing, tightness=100, trim=true) -> (bpm, beat_frames)

Dynamic-programming beat tracker (Ellis 2007; librosa `beat_track`):
the onset envelope is scored against a Gaussian comb of the estimated
period (`bpm` from [`tempo`](@ref) unless given), the best sequence of
beats is found by dynamic programming with a log-lag penalty of weight
`tightness`, and weak leading/trailing beats are dropped when `trim=true`.
Returns the tempo and the beat frame indices; use `get_times(env)[beats]`
for seconds.
"""
function beat_track(env::OnsetStrength; bpm::Maybe{Real}=nothing, tightness::Real=100, trim::Bool=true)
    x = get_data(env)
    all(iszero, x) && return (0.0, Int[])
    fps = get_sr(env) / get_step(env)
    b = isnothing(bpm) ? tempo(env) : Float64(bpm)
    (isfinite(b) && b > 0) || return (b, Int[])
    period = max(1, round(Int, 60 * fps / b))
    score = _local_score(x, period)
    n = length(score)
    backlink = fill(0, n)
    cumscore = zeros(Float64, n)
    lags = collect(-2period:-(round(Int, period / 2)))
    txwt = [-tightness * log(-l / period)^2 for l in lags]
    first_beat = true
    thresh = 0.01 * maximum(score)
    @inbounds for i in 1:n
        best, bestk = -Inf, 0
        for (k, l) in enumerate(lags)
            j = i + l
            c = txwt[k] + (j ≥ 1 ? cumscore[j] : 0.0)
            if j ≥ 1 && c > best
                best, bestk = c, k
            end
        end
        if bestk == 0                       # no reachable predecessor
            cumscore[i] = score[i]
            backlink[i] = 0
        else
            cumscore[i] = score[i] + best
            if first_beat && score[i] < thresh
                backlink[i] = 0
            else
                backlink[i] = i + lags[bestk]
                first_beat = false
            end
        end
    end
    # last beat: the last local maximum of cumscore above half the median of the maxima
    ismax = [i > 1 && i < n && cumscore[i] > cumscore[i-1] && cumscore[i] ≥ cumscore[i+1] for i in 1:n]
    any(ismax) || return (b, Int[])
    med = median(cumscore[ismax])
    lastbeat = findlast(i -> ismax[i] && cumscore[i] ≥ 0.5med, 1:n)
    isnothing(lastbeat) && return (b, Int[])
    beats = Int[lastbeat]
    while backlink[beats[end]] > 0
        push!(beats, backlink[beats[end]])
    end
    reverse!(beats)
    if trim && length(beats) > 1
        boe = score[beats]
        w = hanning(5)
        sm = [sum(boe[max(1, i-2):min(length(boe), i+2)] .* w[max(1, 3-(i-1)):min(5, 5-(i+2-length(boe)))]) for i in 1:length(boe)]
        th = 0.5 * sqrt(mean(sm .^ 2))
        valid = findall(>(th), sm)
        isempty(valid) || (beats = beats[first(valid):last(valid)])
    end
    return (b, beats)
end
