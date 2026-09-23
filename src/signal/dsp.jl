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
