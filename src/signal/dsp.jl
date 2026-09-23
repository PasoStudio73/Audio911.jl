# ---------------------------------------------------------------------------- #
#                                DSP utilities                                 #
# ---------------------------------------------------------------------------- #
# The dsp/ module of audioFlux (MIT licence, Copyright (c) 2023 libAudioFlux):
# Hilbert transform, cross-correlation, convolution, chirp-Z transform.

"""
    hilbert(x) -> Vector{Complex}

Analytic signal of a real signal (audioFlux `hilbert`, MATLAB and SciPy
`hilbert`): `x + i·H{x}`, computed by zeroing the negative frequencies of
the FFT and doubling the positive ones. `real(hilbert(x)) == x`,
`abs(hilbert(x))` is the envelope and `angle` the instantaneous phase.
"""
function hilbert(x::AbstractVector{<:Real})
    T = float(eltype(x))
    N = length(x)
    N == 0 && return Complex{T}[]
    X = fft(Vector{Complex{T}}(x))
    h = zeros(T, N)
    h[1] = 1
    if iseven(N)
        h[N ÷ 2 + 1] = 1
        h[2:N÷2] .= 2
    else
        h[2:(N + 1) ÷ 2] .= 2
    end
    return ifft(X .* h)
end

# ---------------------------------------------------------------------------- #
#                       band-limited (windowed-sinc) resampling                #
# ---------------------------------------------------------------------------- #
# audioFlux's dsp/resample_algorithm.c, which is resampy's `resample_f`
# (Smith's band-limited interpolation) with the same Kaiser filters.

# zero crossings, Kaiser β and roll-off of the three audioFlux/resampy presets
const _SINC_QUALITY = Dict(:best => (64, 14.7696565, 0.9475937),
                           :mid  => (32, 11.6625806, 0.8987969),
                           :fast => (16, 8.5555046, 0.85))

_sinc(v::Real) = abs(v) < 1e-9 ? one(v) : sinpi(v) / (π * v)

# the right half of the windowed ideal low-pass, 2^nbit samples per zero crossing
function _sinc_table(zeros::Int, nbit::Int, rolloff::Real, window::Base.Callable, beta::Real)
    L = zeros * (1 << nbit) + 1
    h = [rolloff * _sinc(rolloff * v) for v in range(0, zeros; length=L)]
    w = window === kaiser ? kaiser(2L - 1; β=beta) : _make_window(Float64, window, 2L - 1, false)
    h .*= view(w, L:2L-1)
    return h
end

function _resample_sinc(x::AbstractVector{T}, ratio::Real; quality::Symbol=:best, nzeros::Maybe{Int}=nothing,
                        rolloff::Maybe{Real}=nothing, window::Base.Callable=kaiser, beta::Maybe{Real}=nothing,
                        nbit::Int=9, scale::Bool=false) where {T<:Real}
    haskey(_SINC_QUALITY, quality) || throw(ArgumentError("quality must be :best, :mid or :fast, got :$quality"))
    ratio > 0 || throw(ArgumentError("the resampling ratio must be positive, got $ratio"))
    z0, b0, r0 = _SINC_QUALITY[quality]
    nz = something(nzeros, z0); ro = something(rolloff, r0); β = something(beta, b0)
    nz > 0 && 0 < ro ≤ 1 || throw(ArgumentError("nzeros must be positive and rolloff in (0, 1]"))
    F = float(T)
    table = _sinc_table(nz, nbit, ro, window, β)
    ratio < 1 && (table .*= ratio)
    L = length(table)
    delta = vcat(diff(table), 0.0)
    bits = 1 << nbit
    scl = min(1.0, Float64(ratio))
    step = floor(Int, scl * bits)
    nin = length(x)
    nout = floor(Int, nin * ratio)
    y = zeros(F, nout)
    @inbounds for i in 0:nout-1
        t = i / ratio
        n = floor(Int, t)
        frac = scl * (t - n)
        fv = frac * bits; off = floor(Int, fv); δ = fv - off
        acc = 0.0
        for j in 0:min(n + 1, (L - off) ÷ step)-1
            k = off + j * step + 1
            acc += (table[k] + δ * delta[k]) * x[n - j + 1]
        end
        frac = scl - frac
        fv = frac * bits; off = floor(Int, fv); δ = fv - off
        for j in 0:min(nin - n - 1, (L - off) ÷ step)-1
            k = off + j * step + 1
            acc += (table[k] + δ * delta[k]) * x[n + j + 2]
        end
        y[i + 1] = acc
    end
    scale && (y ./= sqrt(F(ratio)))
    return y
end

# ---------------------------------------------------------------------------- #
#                               chirp-Z transform                              #
# ---------------------------------------------------------------------------- #
"""
    czt(x, m=length(x), w=cis(-2π/m), a=1) -> Vector{Complex}
    czt(x, band::Tuple{Real,Real}; m=length(x)) -> Vector{Complex}

Chirp-Z transform (Rabiner, Schafer & Rader 1969; MATLAB and SciPy `czt`):
`X[k] = Σₙ x[n] (a w^(-k))^(-n)` for `k = 0 … m-1`, evaluated with
Bluestein's algorithm (three FFTs of a power-of-two length). The defaults
give the DFT.

The second form is audioFlux's `CZT.czt(x, low_w, high_w)`, a zoom FFT:
`m` points evenly spaced from `band[1]` to `band[2]` (excluded), in cycles
per sample (`0 ≤ band[1] < band[2] ≤ 1`; multiply by `sr` for Hz).
"""
function czt(x::AbstractVector{<:Number}, m::Integer=length(x), w::Number=cis(-2π / m), a::Number=1)
    n = length(x)
    n ≥ 1 && m ≥ 1 || throw(ArgumentError("x and m must not be empty"))
    T = float(real(eltype(x)))
    C = Complex{T}
    L = nextpow(2, n + m - 1)
    lw = log(Complex{Float64}(w))
    kk = -(n - 1):max(m, n) - 1
    wk2 = [exp(lw * (Float64(k)^2 / 2)) for k in kk]          # w^(k²/2)
    la = log(Complex{Float64}(a))
    y = zeros(C, L)
    @inbounds for j in 0:n-1
        y[j + 1] = C(x[j + 1] * exp(-la * j) * wk2[n + j])
    end
    v = zeros(C, L)
    @inbounds for j in 1:n+m-1
        v[j] = C(1 / wk2[j])
    end
    g = ifft(fft(y) .* fft(v))
    return C[g[n + k] * wk2[n + k] for k in 0:m-1]
end

function czt(x::AbstractVector{<:Number}, band::Tuple{Real,Real}; m::Integer=length(x))
    lo, hi = band
    0 ≤ lo < hi ≤ 1 || throw(ArgumentError("band must satisfy 0 ≤ low < high ≤ 1 (cycles per sample), got $band"))
    return czt(x, m, cispi(-2 * (hi - lo) / m), cispi(2lo))
end

# ---------------------------------------------------------------------------- #
#                        cross-correlation and convolution                     #
# ---------------------------------------------------------------------------- #
"""
    xcorr(x, y=x; normalize=false) -> Vector

Cross-correlation `r[l] = Σₙ x[n+l] conj(y[n])` for the lags
`l = -(N-1) … N-1` (`2N - 1` values, lag 0 at index `N`), the shorter
input zero-padded to the length `N` of the longer (MATLAB `xcorr`,
audioFlux `Xcorr`), computed by FFT. `normalize=true` divides by
`√(Σ|x|² Σ|y|²)` (MATLAB and audioFlux `coeff`), so the auto-correlation is
1 at lag 0. Real inputs give a real result. See [`autocorrelate`](@ref)
for librosa's non-negative lags.
"""
function xcorr(x::AbstractVector{<:Number}, y::AbstractVector{<:Number}=x; normalize::Bool=false)
    N = max(length(x), length(y))
    N ≥ 1 || throw(ArgumentError("the inputs must not be empty"))
    T = float(real(promote_type(eltype(x), eltype(y))))
    L = nextpow(2, 2N - 1)
    X = zeros(Complex{T}, L); Y = zeros(Complex{T}, L)
    X[1:length(x)] .= x; Y[1:length(y)] .= y
    r = ifft(fft(X) .* conj.(fft(Y)))
    out = vcat(r[L-N+2:L], r[1:N])
    if normalize
        s = sqrt(sum(abs2, x) * sum(abs2, y))
        s > 0 && (out ./= s)
    end
    return eltype(x) <: Real && eltype(y) <: Real ? T.(real.(out)) : out
end

"""
    convolve(a, b; mode=:full) -> Vector

Linear convolution of two vectors (audioFlux `conv`, MATLAB `conv`):
`mode=:full` gives all `length(a) + length(b) - 1` samples, `:same` the
`length(a)` central ones (starting at `floor(length(b)/2)`, as MATLAB and
audioFlux; SciPy's `same` starts one sample earlier for an even `b`) and
`:valid` the `length(a) - length(b) + 1` computed without zero padding.
"""
function convolve(a::AbstractVector{<:Number}, b::AbstractVector{<:Number}; mode::Symbol=:full)
    mode in (:full, :same, :valid) || throw(ArgumentError("mode must be :full, :same or :valid, got :$mode"))
    N, M = length(a), length(b)
    N ≥ 1 && M ≥ 1 || throw(ArgumentError("the inputs must not be empty"))
    T = promote_type(float(eltype(a)), float(eltype(b)))
    full = Vector{T}(DSP.conv(Vector{T}(a), Vector{T}(b)))
    mode === :full && return full
    mode === :same && return full[M ÷ 2 + 1:M ÷ 2 + N]
    return N ≥ M ? full[M:N] : T[]
end
