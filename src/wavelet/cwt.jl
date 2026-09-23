# ---------------------------------------------------------------------------- #
#                            analytic mother wavelets                          #
# ---------------------------------------------------------------------------- #
# Every wavelet is given by its Fourier transform ψ̂(ω) for ω ≥ 0 (rad/sample);
# the transform is zero for negative ω (analytic wavelets).

"""
    morlet(ω; ω0=6)

Fourier transform of the analytic Morlet wavelet,
`π^(-1/4) exp(-(ω - ω0)^2 / 2)` for `ω > 0`. `ω0` sets the time/frequency
trade-off; 6 is the usual choice. Pass `ω -> morlet(ω; ω0=8)` to [`Cwt`](@ref)
to change it.
"""
morlet(ω::Real; ω0::Real=6) = ω > 0 ? π^(-1/4) * exp(-(ω - ω0)^2 / 2) : 0.0

"""
    morse(ω; β=20, γ=3)

Fourier transform of the generalised Morse wavelet, `2 (eγ/β)^(β/γ) ω^β exp(-ω^γ)`
for `ω > 0`, normalised so its peak value is 2 (MATLAB's `cwt` default with
`γ=3`, time-bandwidth `β·γ=60`).
"""
morse(ω::Real; β::Real=20, γ::Real=3) =
    ω > 0 ? 2 * (ℯ * γ / β)^(β / γ) * ω^β * exp(-ω^γ) : 0.0

"""
    bump(ω; μ=5, σ=0.6)

Fourier transform of the bump wavelet, `2 exp(1 - 1/(1 - ((ω-μ)/σ)^2))` on
`|ω - μ| < σ` and zero elsewhere. It is compactly supported in frequency and
gives the sharpest frequency localisation of the three.
"""
function bump(ω::Real; μ::Real=5, σ::Real=0.6)
    u = (ω - μ) / σ
    return abs(u) < 1 ? 2 * exp(1 - 1 / (1 - u^2)) : 0.0
end

# The five wavelets below follow audioFlux (src/filterbank/cwt_filterBank.c,
# MIT licence, Copyright (c) 2023 libAudioFlux): analytic, zero for ω ≤ 0.

"""
    paul(ω; m=4)

Fourier transform of the analytic Paul wavelet of order `m`,
`2^m / sqrt(m (2m-1)!) · ω^m e^(-ω)` for `ω > 0` (Torrence & Compo 1998;
audioFlux `PAUL`). Its equivalent Fourier frequency is `m + 1/2`.
"""
paul(ω::Real; m::Integer=4) = ω > 0 ? _paul_factor(m) * ω^m * exp(-ω) : 0.0
_paul_factor(m::Integer) = Float64(big(2)^m / sqrt(m * factorial(big(2m - 1))))

# Γ(p + 1/2) for integer p ≥ 0
_gamma_half(p::Integer) = Float64(factorial(big(2p)) * sqrt(big(π)) / (big(4)^p * factorial(big(p))))

"""
    dog(ω; m=2, β=2)

Fourier transform of the analytic derivative-of-Gaussian wavelet of even
order `m`, `-(i^m)/sqrt(Γ(m + 1/2)) · ω^m e^(-ω²/β)` for `ω > 0` (real
factor; Torrence & Compo 1998, audioFlux `DOG`). `m = 2` is the Mexican hat.
Its equivalent Fourier frequency is `sqrt(m + 1/2)`.
"""
function dog(ω::Real; m::Integer=2, β::Real=2)
    iseven(m) && m ≥ 2 || throw(ArgumentError("the DOG order must be even and ≥ 2, got $m"))
    ω > 0 || return 0.0
    f = -1 / sqrt(_gamma_half(m))
    isodd(m ÷ 2) && (f = -f)
    return f * ω^m * exp(-ω^2 / β)
end

"""
    mexican(ω; β=2)

Mexican hat wavelet, the [`dog`](@ref) wavelet of order 2 (audioFlux `MEXICAN`).
"""
mexican(ω::Real; β::Real=2) = dog(ω; m=2, β)

"""
    hermit(ω; γ=5, β=2)

audioFlux's Hermitian wavelet, `2/sqrt(γ) π^(-1/4) · (ω-γ)(1+ω-γ) e^(-(ω-γ)²/β)`
for `ω > 0` (`HERMIT`); its centre frequency is taken as `γ + 1`.
"""
hermit(ω::Real; γ::Real=5, β::Real=2) =
    ω > 0 ? 2 / sqrt(γ) * π^(-1/4) * (ω - γ) * (1 + ω - γ) * exp(-(ω - γ)^2 / β) : 0.0

"""
    ricker(ω; γ=4)

Ricker wavelet, `2/sqrt(π) · ω²/γ³ e^(-ω²/γ²)` for `ω > 0` (audioFlux
`RICKER`), peaking at `ω = γ`.
"""
ricker(ω::Real; γ::Real=4) = ω > 0 ? 2 / sqrt(π) * ω^2 / γ^3 * exp(-ω^2 / γ^2) : 0.0

const CWT_WAVELETS = (morse, morlet, bump, paul, dog, mexican, hermit, ricker)

# centre angular frequency (rad/sample at unit scale) that maps a scale to a
# frequency: audioFlux's value for the built-in wavelets, the spectral peak for
# any other function
_centre_omega(ψ) =
    ψ === morse   ? (20 / 3)^(1 / 3) :
    ψ === morlet  ? 6.0 :
    ψ === bump    ? 5.0 :
    ψ === paul    ? 4.5 :
    ψ === dog     ? sqrt(2.5) :
    ψ === mexican ? sqrt(2.5) :
    ψ === hermit  ? 6.0 :
    ψ === ricker  ? 4.0 :
    _peak_omega(ψ)

# peak (centre) angular frequency of a wavelet, found numerically so that any
# user closure works: coarse grid on (0, 64], then golden-section refinement.
function _peak_omega(ψ::Base.Callable)
    grid = range(1e-3, 64, length=4096)
    vals = map(ψ, grid)
    k = argmax(vals)
    a = grid[max(k - 1, 1)]
    b = grid[min(k + 1, length(grid))]
    φ = (sqrt(5) - 1) / 2
    c, d = b - φ * (b - a), a + φ * (b - a)
    for _ in 1:80
        if ψ(c) < ψ(d)
            a = c
        else
            b = d
        end
        c, d = b - φ * (b - a), a + φ * (b - a)
    end
    return (a + b) / 2
end

# time-domain standard deviation of the wavelet at unit scale (in samples),
# estimated from the second moment of |ψ̂|², used to size the edge padding.
function _time_support(ψ::Base.Callable)
    ω = range(1e-3, 64, length=8192)
    p = map(ψ, ω) .^ 2
    s = sum(p)
    s == 0 && return 1.0
    μ = sum(ω .* p) / s
    σω = sqrt(max(sum((ω .- μ) .^ 2 .* p) / s, 1e-12))
    return 1 / (2σω)
end

# ---------------------------------------------------------------------------- #
#                                     info                                     #
# ---------------------------------------------------------------------------- #
struct CwtSetup{T<:AudioData} <: AbstractSetup
    sr        :: Int64
    winsize   :: Int64
    winstep   :: Int64
    wavelet   :: Base.Callable
    voices    :: Int64
    freqrange :: FreqRange
    spectrum  :: Base.Callable
    offset    :: Int64
    scales    :: Vector{T}
    centre    :: Float64
    scale     :: Maybe{Base.Callable}
    pad       :: Int64
end

# ---------------------------------------------------------------------------- #
#                                   Cwt struct                                 #
# ---------------------------------------------------------------------------- #
"""
    Cwt{T} <: AbstractSpectrogram

Continuous wavelet transform scalogram: a `scales × frames` power (or
magnitude) matrix on a geometric frequency grid. It implements the same
front-end interface as [`Stft`](@ref) (`get_spec`, `get_freq`, `get_sr`,
`get_spectrum`, ...), so it can feed `LinSpec`, `MelSpec`, `BarkSpec`,
`ErbSpec`, the cepstra and every descriptor in place of an STFT.

Build one with [`Cwt(frames; kwargs...)`](@ref Cwt(::Frames)) or
[`Cwt(audio; kwargs...)`](@ref Cwt(::AudioFile)).
"""
struct Cwt{T<:AudioData} <: AbstractSpectrogram
    spec   :: Matrix{T}
    freq   :: Vector{T}
    frames :: Frames{T}
    info   :: CwtSetup{T}
end

Base.eltype(::Cwt{T}) where T = T

"""
    get_data(c::Cwt) -> Matrix

The scalogram as stored, `scales × frames` (same as [`get_spec`](@ref)).
"""
@inline get_data(c::Cwt) = c.spec
@inline get_spec(c::Cwt) = c.spec

"""
    get_freq(c::Cwt) -> Vector

Centre frequency of every scale in Hz, ascending.
"""
@inline get_freq(c::Cwt)     = c.freq
@inline get_setup(c::Cwt)    = c.info
@inline get_sr(c::Cwt)       = c.info.sr
@inline get_spectrum(c::Cwt) = c.info.spectrum
@inline get_winsize(c::Cwt)  = c.info.winsize
@inline get_step(c::Cwt)     = c.info.winstep
@inline get_overlap(c::Cwt)  = c.info.winsize - c.info.winstep
@inline get_offset(c::Cwt)   = c.info.offset
@inline get_frames(c::Cwt)   = c.frames
@inline get_parent(c::Cwt)   = c.frames
@inline get_energy(c::Cwt)   = get_energy(c.frames)

"""
    get_scales(c::Cwt) -> Vector

The wavelet scales (in samples) matching `get_freq(c)`.
"""
@inline get_scales(c::Cwt) = c.info.scales

function Base.show(io::IO, c::Cwt{T}) where T
    nsc, nfr = size(c.spec)
    print(io, "Cwt{$T}($nfr frames × $nsc scales, $(c.info.wavelet), sr=$(c.info.sr) Hz)")
end

function Base.show(io::IO, ::MIME"text/plain", c::Cwt{T}) where T
    nsc, nfr = size(c.spec)
    println(io, "Cwt{$T}")
    println(io, "    Sample rate:     $(c.info.sr) Hz")
    println(io, "    Frames:          $nfr")
    println(io, "    Scales:          $nsc ", isnothing(c.info.scale) ? "($(c.info.voices) voices per octave)" : "($(nameof(c.info.scale)) grid)")
    println(io, "    Frequency range: $(round(c.freq[1], digits=1)) - $(round(c.freq[end], digits=1)) Hz")
    println(io, "    Wavelet:         $(c.info.wavelet)")
    println(io, "    Window size:     $(c.info.winsize) samples")
    println(io, "    Hop size:        $(c.info.winstep) samples")
    print(io,   "    Spectrum type:   $(c.info.spectrum)")
end

# ---------------------------------------------------------------------------- #
#                                  computation                                 #
# ---------------------------------------------------------------------------- #
function _reflect_pad(x::Vector{T}, pad::Int) where T
    n = length(x)
    pad = min(pad, n - 1)
    y = Vector{T}(undef, n + 2pad)
    copyto!(y, pad + 1, x, 1, n)
    @inbounds for i in 1:pad
        y[pad + 1 - i] = x[i + 1]
        y[n + pad + i] = x[n - i]
    end
    return y, pad
end

# frequency grid of a transform: geometric `voices` per octave between the
# ends of `freqrange`, or the band centres of a filterbank scale
function _cwt_grid(::Type{T}, scale, freqrange::FreqRange, nbands::Int, voices::Int,
                   bins_per_octave::Int) where T
    if isnothing(scale)
        fmin, fmax = get_low(freqrange), get_hi(freqrange)
        n = floor(Int, log2(fmax / fmin) * voices) + 1
        return T[fmin * 2.0^((k - 1) / voices) for k in 1:n]
    end
    scale in AVAIL_SCALES || throw(ArgumentError("scale must be one of $(AVAIL_SCALES), got $scale"))
    return Vector{T}(_band_edges(T, scale, freqrange, nbands, bins_per_octave)[2:end-1])
end

# The per-band core shared by every wavelet stage: the padded signal is
# transformed once; for every band the wavelet (and, with `deriv`, the
# wavelet times iω, whose inverse transform is the time derivative of the
# coefficients) is applied and inverse-transformed. `g(chunk, k, w, dw)` is
# called with the band series (the unpadded part, a view) from the thread
# that owns chunk `chunk`; `g` may write row `k` of a shared matrix or a
# per-chunk accumulator.
function _cwt_bands(g, xp::Vector{T}, pad::Int, N::Int, freq::AbstractVector, sr::Int,
                    wavelet::Base.Callable, centre::Real; deriv::Bool=false) where T
    M = length(xp)
    X = rfft(xp)
    K = length(X)
    ω = T[(k - 1) * (2π / M) for k in 1:K]
    plan = plan_ifft!(zeros(Complex{T}, M))
    chunks = _chunks(length(freq))
    Threads.@threads for c in eachindex(chunks)
        buf  = zeros(Complex{T}, M)
        dbuf = deriv ? zeros(Complex{T}, M) : buf
        for k in chunks[c]
            s = T(centre * sr / (2π * freq[k]))
            @inbounds for i in 1:K
                h = X[i] * T(wavelet(s * ω[i]))
                buf[i] = h
                deriv && (dbuf[i] = h * Complex{T}(0, ω[i]))
            end
            @inbounds fill!(view(buf, K+1:M), zero(Complex{T}))
            plan * buf
            if deriv
                @inbounds fill!(view(dbuf, K+1:M), zero(Complex{T}))
                plan * dbuf
            end
            g(c, k, view(buf, pad+1:pad+N), deriv ? view(dbuf, pad+1:pad+N) : nothing)
        end
    end
    return length(chunks)
end

# symmetric padding (edge repeated) by `p` samples, audioFlux's is_padding
function _symmetric_pad(x::AbstractVector{T}, p::Int) where T
    n = length(x)
    p = min(p, n)
    return vcat(x[p:-1:1], x, x[n:-1:n-p+1])
end

"""
    cwt(x, sr; wavelet=morse, nbands=84, scale=octave, freqrange, bins_per_octave=12,
        centre, pad=true, T=eltype(x)) -> (W, freq)

Complex continuous wavelet transform of a whole signal (audioFlux `CWT`):
`W` is `nbands × length(x)`, row `k` the coefficients at the centre
frequency `freq[k]` (ascending). Every band multiplies the FFT of the
signal by the analytic wavelet `ψ̂(s ω)` at the scale `s = centre · sr /
(2π f)` (L1 normalisation) and inverse transforms it.

- `wavelet`: [`morse`](@ref), [`morlet`](@ref), [`bump`](@ref),
  [`paul`](@ref), [`dog`](@ref), [`mexican`](@ref), [`hermit`](@ref),
  [`ricker`](@ref), or any function of the angular frequency
- `scale`, `nbands`, `freqrange`, `bins_per_octave`: the frequency grid, as
  in [`auditory_fbank`](@ref) (`octave` from C1 by default); `scale=nothing`
  with `voices` gives the geometric grid of [`Cwt`](@ref)
- `centre`: the angular frequency of the wavelet at unit scale (defaults to
  audioFlux's value for the built-in wavelets, the spectral peak otherwise)
- `pad=true`: symmetric padding by half the signal length on both sides
  (audioFlux `is_padding`); `false` treats the signal as periodic
"""
function cwt(x::AbstractVector{<:Real}, sr::Int; wavelet::Base.Callable=morse, nbands::Int=84,
             scale::Maybe{Base.Callable}=octave, voices::Int=12,
             freqrange::FreqRange=(scale in (octave, logspace) || isnothing(scale) ? 33 : 0, sr ÷ 2),
             bins_per_octave::Int=12, centre::Real=_centre_omega(wavelet), pad::Bool=true,
             T::Type=float(eltype(x)))
    N = length(x)
    N ≥ 2 || throw(ArgumentError("the signal needs at least two samples"))
    freq = _cwt_grid(T, scale, freqrange, nbands, voices, bins_per_octave)
    freq[end] < sr / 2 * (1 + 4eps(T)) || throw(ArgumentError(
        "the highest band ($(round(freq[end], digits=1)) Hz) is above the Nyquist frequency"))
    p  = pad ? N ÷ 2 : 0
    xp = pad ? _symmetric_pad(Vector{T}(x), p) : Vector{T}(x)
    W  = Matrix{Complex{T}}(undef, length(freq), N)
    _cwt_bands(xp, p, N, freq, sr, wavelet, centre) do _, k, w, _
        @inbounds W[k, :] .= w
    end
    return W, freq
end

"""
    Cwt(frames::Frames; kwargs...) -> Cwt

Compute a wavelet scalogram on the time grid of `frames`.

The signal is analysed with an analytic wavelet, one scale at a time, in the
frequency domain (one inverse FFT per scale, L1 normalisation). The squared
modulus (or the modulus, for `spectrum=magnitude`) is then pooled over every
frame of `frames`, weighted by the frame window, so the result has one
column per frame exactly like `Stft(frames)` and can replace it anywhere in
the pipeline. DC removal and pre-emphasis options of the frames are ignored:
the transform is applied to the raw signal.

# Keyword Arguments
- `wavelet::Base.Callable=morlet`: `morlet`, `morse`, `bump`, `paul`, `dog`,
  `mexican`, `hermit`, `ricker`, or any function `ω -> ψ̂(ω)` of the angular
  frequency in rad/sample
- `voices::Int=12`: scales per octave of the default geometric grid
- `freqrange::FreqRange=(sr ÷ winsize, sr ÷ 2)`: lowest and highest centre
  frequency in Hz; scales are spaced geometrically between them
- `scale=nothing`: a filterbank scale (`octave`, `htk`, `bark`, `erb`,
  `linspace`, `logspace`, ...) with `nbands` and `bins_per_octave` puts the
  bands on audioFlux's grid instead
- `centre`: the wavelet's angular frequency at unit scale (see [`cwt`](@ref))
- `spectrum::Base.Callable=power`: `power` (`|W|²`) or `magnitude` (`|W|`)

# Examples
```julia
frames = Frames(audio; winsize=512, winstep=256, type=rect)
cwt    = Cwt(frames; voices=16, freqrange=(50, 8000))
mel    = MelSpec(cwt; nbands=26)     # wavelet-fed mel spectrogram
mfcc   = Mfcc(mel; ncoeffs=13)
cwt_p  = Cwt(frames; wavelet=paul, scale=octave, nbands=84, freqrange=(33, 8000))
```
"""
function Cwt(
    frames    :: Frames{T};
    wavelet   :: Base.Callable=morlet,
    voices    :: Int64=12,
    freqrange :: FreqRange=(max(1, get_sr(frames) ÷ get_winsize(frames)), get_sr(frames) ÷ 2),
    spectrum  :: Base.Callable=power,
    scale     :: Maybe{Base.Callable}=nothing,
    nbands    :: Int64=84,
    bins_per_octave :: Int64=12,
    centre    :: Real=_centre_omega(wavelet),
) where {T<:AudioData}
    sr = get_sr(frames)
    x  = get_signal(frames)
    N  = length(x)
    fmin, fmax = get_low(freqrange), get_hi(freqrange)
    if isnothing(scale)
        0 < fmin < fmax ≤ sr ÷ 2 || throw(ArgumentError(
            "freqrange must satisfy 0 < fmin < fmax ≤ sr/2, got $freqrange with sr = $sr"))
        voices ≥ 1 || throw(ArgumentError("voices must be ≥ 1, got $voices"))
    end
    spectrum in (power, magnitude) ||
        throw(ArgumentError("spectrum must be `power` or `magnitude`, got $spectrum"))

    freq    = _cwt_grid(T, scale, freqrange, nbands, voices, bins_per_octave)
    freq[end] ≤ sr / 2 * (1 + 4eps(T)) || throw(ArgumentError(
        "the highest band ($(round(freq[end], digits=1)) Hz) is above the Nyquist frequency"))
    nscales = length(freq)
    scales  = T[centre * sr / (2π * f) for f in freq]

    # reflect-pad against circular wrap-around by four time supports
    support = _time_support(wavelet)
    xp, pad = _reflect_pad(x, ceil(Int, 4 * support * maximum(scales)))

    w    = get_window(frames)
    ws   = length(w)
    wsum = sum(w)
    spec = Matrix{T}(undef, nscales, length(frames))
    _cwt_bands(xp, pad, N, freq, sr, wavelet, centre) do _, si, y, _
        @inbounds for (j, st) in enumerate(frames.starts)
            base = st - 1
            acc = zero(T)
            @simd for i in 1:ws
                acc += w[i] * spectrum(y[base + i])
            end
            spec[si, j] = acc / wsum
        end
    end

    info = CwtSetup{T}(sr, get_winsize(frames), get_step(frames), wavelet, voices,
                       freqrange, spectrum, get_offset(frames), scales, Float64(centre), scale, pad)
    return Cwt{T}(spec, freq, frames, info)
end

"""
    get_complex(c::Cwt) -> Matrix{Complex}

The complex wavelet coefficients at every frame centre, `scales × frames`
(recomputed from the frames).
"""
function get_complex(c::Cwt{T}) where T
    i = c.info
    x = get_signal(c.frames)
    xp, pad = _reflect_pad(x, i.pad)
    cen = _frame_centres(c.frames)
    out = Matrix{Complex{T}}(undef, length(c.freq), length(cen))
    _cwt_bands(xp, pad, length(x), c.freq, i.sr, i.wavelet, i.centre) do _, k, y, _
        @inbounds for (j, n) in enumerate(cen)
            out[k, j] = y[n]
        end
    end
    return out
end

"""
    Cwt(audio::AudioFile; kwargs...) -> Cwt
    Cwt(x::AbstractVecOrMat, sr::Int; kwargs...) -> Cwt

Frame the signal (rectangular pooling window by default) and compute its
scalogram. Framing keywords (`winsize`, `winstep`, `type`, `periodic`,
`center`, `pad_mode`) go to [`Frames`](@ref); the rest to
[`Cwt(::Frames)`](@ref).

```julia
cwt = Cwt(audio; winsize=512, winstep=256, wavelet=morse, voices=10)
```
"""
function Cwt(
    audio    :: AbstractVecOrMat{<:Real},
    sr       :: Int64;
    winsize  :: Int64=sr ≤ 8000 ? 256 : 512,
    winstep  :: Int64=winsize ÷ 2,
    win      :: Maybe{NamedTuple}=nothing,
    type     :: Base.Callable=rect,
    periodic :: Bool=true,
    center   :: Bool=false,
    pad_mode :: Symbol=:constant,
    kwargs...
)
    frames = Frames(audio, sr; winsize, winstep, win, type, periodic, center, pad_mode)
    return Cwt(frames; kwargs...)
end

Cwt(a::AudioFile; kwargs...) = Cwt(get_data(a), get_sr(a); kwargs...)
