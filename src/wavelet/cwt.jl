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
    println(io, "    Scales:          $nsc ($(c.info.voices) voices per octave)")
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
- `wavelet::Base.Callable=morlet`: `morlet`, `morse`, `bump`, or any function
  `ω -> ψ̂(ω)` of the angular frequency in rad/sample
- `voices::Int=12`: scales per octave
- `freqrange::FreqRange=(sr ÷ winsize, sr ÷ 2)`: lowest and highest centre
  frequency in Hz; scales are spaced geometrically between them
- `spectrum::Base.Callable=power`: `power` (`|W|²`) or `magnitude` (`|W|`)

# Examples
```julia
frames = Frames(audio; winsize=512, winstep=256, type=rect)
cwt    = Cwt(frames; voices=16, freqrange=(50, 8000))
mel    = MelSpec(cwt; nbands=26)     # wavelet-fed mel spectrogram
mfcc   = Mfcc(mel; ncoeffs=13)
```
"""
function Cwt(
    frames    :: Frames{T};
    wavelet   :: Base.Callable=morlet,
    voices    :: Int64=12,
    freqrange :: FreqRange=(max(1, get_sr(frames) ÷ get_winsize(frames)), get_sr(frames) ÷ 2),
    spectrum  :: Base.Callable=power,
) where {T<:AudioData}
    sr = get_sr(frames)
    x  = get_signal(frames)
    N  = length(x)
    fmin, fmax = get_low(freqrange), get_hi(freqrange)
    0 < fmin < fmax ≤ sr ÷ 2 || throw(ArgumentError(
        "freqrange must satisfy 0 < fmin < fmax ≤ sr/2, got $freqrange with sr = $sr"))
    voices ≥ 1 || throw(ArgumentError("voices must be ≥ 1, got $voices"))
    spectrum in (power, magnitude) ||
        throw(ArgumentError("spectrum must be `power` or `magnitude`, got $spectrum"))

    # geometric frequency grid, ascending
    nscales = floor(Int, log2(fmax / fmin) * voices) + 1
    freq    = T[fmin * 2.0^((k - 1) / voices) for k in 1:nscales]
    ωp      = _peak_omega(wavelet)
    scales  = T[ωp * sr / (2π * f) for f in freq]

    # reflect-pad against circular wrap-around, transform the padded signal once
    support = _time_support(wavelet)
    xp, pad = _reflect_pad(x, ceil(Int, 4 * support * maximum(scales)))
    M   = length(xp)
    X   = rfft(xp)
    K   = length(X)
    ω   = (0:K-1) .* (2π / M)
    plan = plan_ifft!(zeros(Complex{T}, M))

    w    = get_window(frames)
    ws   = length(w)
    wsum = sum(w)
    nfr  = length(frames)
    spec = Matrix{T}(undef, nscales, nfr)

    Threads.@threads for chunk in _chunks(nscales)
        buf = zeros(Complex{T}, M)
        for si in chunk
            s = scales[si]
            @inbounds for k in 1:K
                buf[k] = X[k] * T(wavelet(s * ω[k]))
            end
            @inbounds fill!(view(buf, K+1:M), zero(Complex{T}))
            plan * buf
            @inbounds for (j, st) in enumerate(frames.starts)
                base = st + pad - 1
                acc = zero(T)
                @simd for i in 1:ws
                    acc += w[i] * spectrum(buf[base + i])
                end
                spec[si, j] = acc / wsum
            end
        end
    end

    info = CwtSetup{T}(sr, get_winsize(frames), get_step(frames), wavelet, voices,
                       freqrange, spectrum, get_offset(frames), scales)
    return Cwt{T}(spec, freq, frames, info)
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
