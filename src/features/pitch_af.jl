# ---------------------------------------------------------------------------- #
#                       spectral pitch estimators (audioFlux)                  #
# ---------------------------------------------------------------------------- #
# Ports of audioFlux's PitchPEF, PitchHPS, PitchLHS and PitchSTFT
# (src/mir/_pitch_*.c, MIT licence, Copyright (c) 2023 libAudioFlux). Every
# method estimates the fundamental of one raw frame and plugs into
# `Pitch(frames; method=...)`. The frame is multiplied by `window` (a
# periodic window of the frame length, audioFlux's default in brackets).

_af_window(::Type{T}, window, n) where T = window === rect ? ones(T, n) : _make_window(T, window, n, true)

# nearest power of two to v (audioFlux util_roundPowerTwo)
function _round_pow2(v::Integer)
    ispow2(v) && return Int(v)
    lo = prevpow(2, v); hi = nextpow(2, v)
    return v - lo < hi - v ? lo : hi
end

# magnitude spectrum (full length) of a windowed, zero-padded frame
function _frame_mag(x::AbstractVector{T}, window, nfft::Int) where T
    w = _af_window(T, window, length(x))
    buf = zeros(T, nfft)
    @inbounds for i in eachindex(x)
        buf[i] = x[i] * w[i]
    end
    return abs.(fft(buf))
end

# bins of a spectrum of `nfft` points at `sr` whose frequency lies in `range`
_bin_range(range::FreqRange, sr::Int, nfft::Int) =
    ceil(Int, get_low(range) * nfft / sr):floor(Int, get_hi(range) * nfft / sr)

"""
    pitch_hps(x, sr; range=(50, 400), threshold=0, harmonics=5, window=hamming) -> f0

Harmonic product spectrum (Schroeder 1968; audioFlux `PitchHPS`): the
windowed frame is zero-padded to the power of two nearest to `sr` (about
1 Hz per bin), and the product `Π_{k=1}^{H} |X[k·j]|` over `harmonics`
multiples of every candidate bin `j` in `range` is maximised. `threshold`
is unused (the method always returns a frequency).
"""
function pitch_hps(x::AbstractVector{T}, sr::Int; range::FreqRange=(50, 400), threshold::Real=0,
                   harmonics::Int=5, window::Base.Callable=hamming) where T
    nfft = max(_round_pow2(sr), nextpow(2, length(x)))
    A = _frame_mag(x, window, nfft)
    H = clamp(harmonics, 1, max(1, sr ÷ (get_hi(range) + 1)))
    best, bj = -one(T), 0
    for j in _bin_range(range, sr, nfft)
        p = one(T)
        for k in 1:H
            p *= A[mod(j * k, nfft) + 1]
        end
        p > best && ((best, bj) = (p, j))
    end
    return T(bj * sr / nfft)
end

"""
    pitch_lhs(x, sr; range=(50, 400), threshold=0, harmonics=5, window=hamming) -> f0

Log-harmonic summation (Hermes 1988; MATLAB and audioFlux `LHS`): as
[`pitch_hps`](@ref) with the sum of the log magnitudes `Σ log |X[k·j]|`.
"""
function pitch_lhs(x::AbstractVector{T}, sr::Int; range::FreqRange=(50, 400), threshold::Real=0,
                   harmonics::Int=5, window::Base.Callable=hamming) where T
    nfft = max(_round_pow2(sr), nextpow(2, length(x)))
    L = log.(max.(_frame_mag(x, window, nfft), floatmin(T)))
    H = clamp(harmonics, 1, max(1, sr ÷ (get_hi(range) + 1)))
    best, bj = T(-Inf), 0
    for j in _bin_range(range, sr, nfft)
        s = zero(T)
        for k in 1:H
            s += L[mod(j * k, nfft) + 1]
        end
        s > best && ((best, bj) = (s, j))
    end
    return T(bj * sr / nfft)
end

"""
    pitch_pef(x, sr; range=(50, 400), threshold=0, cutoff=4000, alpha=10, beta=0.5,
              gamma=1.8, window=hamming) -> f0

Pitch estimation filter (Gonzalez & Brookes 2011; MATLAB and audioFlux
`PEF`). The power spectrum of the windowed frame (FFT of twice its length)
is interpolated on `2N` log-spaced frequencies from 10 Hz to `cutoff` and
weighted by the log-grid spacing; it is cross-correlated with the filter
`1/(γ - cos 2πq) - mean` over `N` log-spaced `q` from `β` to `α + β`,
whose peaks sit on the harmonics; the lag of the correlation maximum within
`range` is the fundamental on the log grid.
"""
function pitch_pef(x::AbstractVector{T}, sr::Int; range::FreqRange=(50, 400), threshold::Real=0,
                   cutoff::Real=4000, alpha::Real=10, beta::Real=0.5, gamma::Real=1.8,
                   window::Base.Callable=hamming) where T
    N = length(x)
    fl, fh = T(get_low(range)), T(get_hi(range))
    cut = min(T(max(cutoff, fh)), T(sr / 2 - 1))
    w = _af_window(T, window, N)
    buf = zeros(T, 2N)
    @inbounds for i in 1:N; buf[i] = x[i] * w[i]; end
    X = rfft(buf)
    P = abs2.(X)                                   # N+1 bins, 0 … sr/2
    lin = T(sr / 2) / N
    logf = T[exp10(v) for v in LinRange(1, log10(cut), 2N)]
    # linear interpolation of the power spectrum on the log grid
    Pi = Vector{T}(undef, 2N)
    @inbounds for (i, f) in enumerate(logf)
        u = f / lin
        k = min(floor(Int, u), N - 1)
        t = u - k
        Pi[i] = (1 - t) * P[k + 1] + t * P[k + 2]
    end
    bw = similar(Pi)
    @inbounds for i in 2:2N-1
        bw[i] = (logf[i + 1] - logf[i - 1]) / (4N)
    end
    bw[1] = bw[2]; bw[2N] = bw[2N - 1]
    Pi .*= bw
    # the estimation filter over N log-spaced q
    q = T[exp10(v) for v in LinRange(log10(beta), log10(alpha + beta), N)]
    h = T[1 / (gamma - cospi(2qi)) for qi in q]
    edges = vcat(q[1], [(q[i - 1] + q[i]) / 2 for i in 2:N], q[N])
    d = diff(edges)
    filt = h .- sum(d .* h) / sum(d)
    pad = count(<(1), q)
    # log-grid indices nearest to the frequency bounds
    near(f) = argmin(abs.(logf .- f)) - 1
    imin, imax = near(fl), near(fh)
    row(n) = n < pad ? zero(T) : (n - pad < 2N ? Pi[n - pad + 1] : zero(T))
    best, blag = T(-Inf), imin
    for lag in imin:imax
        r = zero(T)
        @inbounds for p in 0:N-1
            r += row(p + lag) * filt[p + 1]
        end
        r > best && ((best, blag) = (r, lag))
    end
    return logf[blag + 1]
end

"""
    pitch_stft(x, sr; range=(50, 400), threshold=20, window=hamming, tolerance=0.03) -> f0

Spectral-peak pitch with a harmonic sieve (after audioFlux `PitchSTFT`): the
local maxima of the windowed frame's power spectrum within `threshold` dB of
the strongest one are located with quadratic interpolation on the log
magnitude; every peak in `range`, and its halves and thirds, is a candidate
fundamental, scored by the dB-weighted number of peaks lying within
`tolerance` (relative) of one of its harmonics. The best-scoring candidate
is returned, 0 when no peak exists. audioFlux's own voting heuristics
(`trist`) are not reproduced.
"""
function pitch_stft(x::AbstractVector{T}, sr::Int; range::FreqRange=(50, 400), threshold::Real=20,
                    window::Base.Callable=hamming, tolerance::Real=0.03) where T
    N = length(x)
    nfft = nextpow(2, N)
    A = _frame_mag(x, window, nfft)
    L = 20 .* log10.(max.(A[1:nfft÷2+1], floatmin(T)))
    top = maximum(L)
    pf = T[]; pd = T[]
    kmax = min(nfft ÷ 2, ceil(Int, 10 * get_hi(range) * nfft / sr))
    @inbounds for k in 2:kmax
        if L[k] > L[k - 1] && L[k] ≥ L[k + 1] && L[k] > top - threshold
            a, b, c = L[k - 1], L[k], L[k + 1]
            den = a - 2b + c
            δ = den == 0 ? zero(T) : (a - c) / (2den)
            push!(pf, (k - 1 + δ) * sr / nfft)
            push!(pd, b - (a - c) * δ / 4 - (top - threshold))
        end
    end
    isempty(pf) && return zero(T)
    fl, fh = get_low(range), get_hi(range)
    best, bf = -one(T), zero(T)
    for f in pf, div in 1:3
        c = f / div
        fl ≤ c ≤ fh || continue
        s = zero(T)
        for (g, d) in zip(pf, pd)
            h = round(g / c)
            h ≥ 1 && abs(g - h * c) ≤ tolerance * h * c && (s += d)
        end
        s > best && ((best, bf) = (s, c))
    end
    return bf
end
