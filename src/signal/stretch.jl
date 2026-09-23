# ---------------------------------------------------------------------------- #
#                    phase vocoder, time stretch, pitch shift                  #
# ---------------------------------------------------------------------------- #
# Ports of audioFlux's dsp/phase_vocoder.c, mir/timeStretch_algorithm.c and
# mir/pitchShift_algorithm.c (MIT licence, Copyright (c) 2023 libAudioFlux).

"""
    phase_vocoder(C, rate; hop) -> Matrix{Complex}

Time-scale a one-sided complex STFT `C` (`bins × frames`, `nfft ÷ 2 + 1`
bins, hop `hop`) by `rate` (Flanagan & Golden 1966; librosa
`phase_vocoder`, audioFlux `phase_vocoder`): output frame `i` sits at the
fractional input frame `t = i · rate`, its magnitude interpolates linearly
between frames `⌊t⌋` and `⌊t⌋ + 1` (zeros past the end), and its phase
accumulates the expected advance `2π k · hop / nfft` of every bin plus the
wrapped measured deviation. `rate > 1` speeds up (fewer frames,
`ceil(frames / rate)`), `rate < 1` slows down.
"""
function phase_vocoder(C::AbstractMatrix{Complex{T}}, rate::Real; hop::Int) where {T<:AbstractFloat}
    rate > 0 || throw(ArgumentError("rate must be positive, got $rate"))
    hop > 0 || throw(ArgumentError("hop must be positive, got $hop"))
    nb, nf = size(C)
    nout = ceil(Int, nf / rate)
    phi = T.(range(0, π * hop; length=nb))
    ph = T[angle(C[k, 1]) for k in 1:nb]
    D = Matrix{Complex{T}}(undef, nb, nout)
    r = T(rate)
    @inbounds for i in 0:nout-1
        t = i * r
        k = floor(Int, t)
        α = t - k
        for b in 1:nb
            c1 = k < nf ? C[b, k + 1] : zero(Complex{T})
            c2 = k + 1 < nf ? C[b, k + 2] : zero(Complex{T})
            D[b, i + 1] = ((1 - α) * abs(c1) + α * abs(c2)) * cis(ph[b])
            dp = angle(c2) - angle(c1) - phi[b]
            dp -= 2T(π) * round(dp / 2T(π), RoundNearestTiesAway)
            ph[b] += phi[b] + dp
        end
    end
    return D
end

"""
    time_stretch(x, rate; winsize=4096, winstep=winsize ÷ 4, window=hanning) -> Vector
    time_stretch(audio::AudioFile, rate; kwargs...) -> AudioFile

Change the duration of a signal by `1 / rate` without changing its pitch
(audioFlux `TimeStretch`, librosa `effects.time_stretch`): the complex STFT
(uncentred frames, periodic `window`) goes through
[`phase_vocoder`](@ref) and back through the weighted overlap-add
[`istft`](@ref). The result has `round(length(x) / rate)` samples.
"""
function time_stretch(x::AbstractVector{<:Real}, rate::Real; winsize::Int=4096, winstep::Int=winsize ÷ 4,
                      window::Base.Callable=hanning)
    rate > 0 || throw(ArgumentError("rate must be positive, got $rate"))
    T = float(eltype(x))
    fr = Frames(Vector{T}(x), 1; winsize, winstep, type=window)
    D = phase_vocoder(get_complex(Stft(fr)), rate; hop=winstep)
    n = round(Int, length(x) / rate, RoundNearestTiesAway)
    return istft(D, winsize, winstep; window, method=:wola, length=n)
end

"""
    pitch_shift(x, n_steps; winsize=4096, winstep=winsize ÷ 4, window=hanning, quality=:fast) -> Vector
    pitch_shift(audio::AudioFile, n_steps; kwargs...) -> AudioFile

Shift the pitch of a signal by `n_steps` semitones keeping its duration
(audioFlux `PitchShift`, librosa `effects.pitch_shift`): the signal is
time-stretched by `rate = 2^(-n_steps / 12)` with [`time_stretch`](@ref),
then resampled by `rate` with the band-limited sinc interpolator
(`resample(...; method=:sinc)`, audioFlux's `:fast` quality and its `√rate`
scaling), and cut or zero-padded to the original length.
"""
function pitch_shift(x::AbstractVector{<:Real}, n_steps::Real; winsize::Int=4096, winstep::Int=winsize ÷ 4,
                     window::Base.Callable=hanning, quality::Symbol=:fast)
    T = float(eltype(x))
    rate = exp2(-n_steps / 12)
    y = time_stretch(x, rate; winsize, winstep, window)
    z = _resample_sinc(y, rate; quality, scale=true)
    n = length(x)
    return length(z) ≥ n ? T.(z[1:n]) : vcat(T.(z), zeros(T, n - length(z)))
end

for f in (:time_stretch, :pitch_shift)
    @eval function $f(a::AudioFile, v::Real; kwargs...)
        X = get_data(a)
        Y = reduce(hcat, [$f(c, v; kwargs...) for c in eachcol(X)])
        return AudioFile(eltype(X).(Y), get_sr(a); mono=false)
    end
end
