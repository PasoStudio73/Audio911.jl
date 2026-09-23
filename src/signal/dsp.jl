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
