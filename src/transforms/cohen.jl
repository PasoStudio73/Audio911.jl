# ---------------------------------------------------------------------------- #
#                 Cohen-class distributions: Wigner-Ville, Choi-Williams       #
# ---------------------------------------------------------------------------- #
# audioFlux declares WVD and CWD in include/wvd_algorithm.h and
# include/cwd_algorithm.h without an implementation; these follow the
# published discrete definitions (Claasen & Mecklenbräuker 1980; Choi &
# Williams 1989; Auger et al., Time-Frequency Toolbox `tfrpwv`, `tfrcw`),
# computed on the analytic signal at the centre of every frame.

"""
    wvd(x; nfft=nextpow(2, length(x))) -> Matrix

Wigner-Ville distribution of a whole (short) signal, computed on its
analytic signal `z` ([`hilbert`](@ref)): column `n` is the FFT over the lag
`m` of `z[n+m] conj(z[n-m])`, row `k` the frequency `(k-1)·sr/(2 nfft)`.
The result is real, `nfft × length(x)`, and can be negative (interference
terms). The cost is `O(N · nfft)`; use [`Wvd`](@ref) for long signals.
"""
function wvd(x::AbstractVector{<:Real}; nfft::Int=nextpow(2, length(x)))
    z = hilbert(x)
    T = real(eltype(z))
    N = length(z)
    out = Matrix{T}(undef, nfft, N)
    plan = plan_fft!(zeros(Complex{T}, nfft))
    buf = zeros(Complex{T}, nfft)
    for n in 1:N
        fill!(buf, zero(Complex{T}))
        M = min(n - 1, N - n, nfft ÷ 2 - 1)
        @inbounds for m in -M:M
            buf[mod(m, nfft) + 1] = z[n + m] * conj(z[n - m])
        end
        plan * buf
        @inbounds for k in 1:nfft
            out[k, n] = real(buf[k])
        end
    end
    return out
end

struct CohenSetup <: AbstractSetup
    sr       :: Int64
    winsize  :: Int64
    winstep  :: Int64
    offset   :: Int64
    kernel   :: Symbol          # :wvd or :cwd
    sigma    :: Float64
    smooth   :: Int64
end

"""
    CohenDistribution{T} <: AbstractSpectrogram

A time-frequency distribution of Cohen's class evaluated at the centre of
every frame of a [`Frames`](@ref) object, built by [`Wvd`](@ref) or
[`Cwd`](@ref). `get_spec` is the distribution clipped at zero (the
interface requires non-negative values); [`get_distribution`](@ref) is the
signed distribution.
"""
struct CohenDistribution{T<:AudioData} <: AbstractSpectrogram
    spec   :: Matrix{T}
    dist   :: Matrix{T}
    freq   :: Vector{T}
    frames :: Frames{T}
    info   :: CohenSetup
end
Base.eltype(::CohenDistribution{T}) where T = T
get_data(c::CohenDistribution)     = c.spec
get_spec(c::CohenDistribution)     = c.spec
get_freq(c::CohenDistribution)     = c.freq
get_setup(c::CohenDistribution)    = c.info
get_sr(c::CohenDistribution)       = c.info.sr
get_spectrum(c::CohenDistribution) = power
get_winsize(c::CohenDistribution)  = c.info.winsize
get_step(c::CohenDistribution)     = c.info.winstep
get_offset(c::CohenDistribution)   = c.info.offset
get_frames(c::CohenDistribution)   = c.frames
get_parent(c::CohenDistribution)   = c.frames
get_energy(c::CohenDistribution)   = get_energy(c.frames)

"""
    get_distribution(c::CohenDistribution) -> Matrix

The signed distribution, `bins × frames` (the Wigner-Ville distribution
takes negative values where components interfere).
"""
get_distribution(c::CohenDistribution) = c.dist

Base.show(io::IO, c::CohenDistribution{T}) where T =
    print(io, "CohenDistribution{$T}(:$(c.info.kernel), $(size(c.spec, 2)) frames × $(size(c.spec, 1)) bins)")

function _cohen(frames::Frames{T}, kernel::Symbol, sigma::Real, smooth::Int) where T
    sr = get_sr(frames)
    z  = Vector{Complex{T}}(hilbert(get_signal(frames)))
    N  = length(z)
    w  = get_window(frames)
    L  = length(w)
    nfft = L
    M  = L ÷ 2 - 1                       # largest lag
    cen = _frame_centres(frames)
    D  = Matrix{T}(undef, nfft, length(cen))
    plan = plan_fft!(zeros(Complex{T}, nfft))
    # lag window h[m] from the frame window, centred on lag 0
    h = T[w[clamp(L ÷ 2 + 1 + m, 1, L)] for m in -M:M]
    # Choi-Williams time-smoothing weights for every lag
    U = smooth
    G = kernel === :cwd ?
        [m == 0 ? T[μ == 0 for μ in -U:U] :
         (g = T[exp(-T(sigma) * μ^2 / (4 * T(m)^2)) for μ in -U:U]; g ./ sum(g)) for m in -M:M] :
        nothing
    chunks = _chunks(length(cen))
    Threads.@threads for c in eachindex(chunks)
        buf = zeros(Complex{T}, nfft)
        for j in chunks[c]
            n = cen[j]
            fill!(buf, zero(Complex{T}))
            @inbounds for (im, m) in enumerate(-M:M)
                if kernel === :wvd
                    (1 ≤ n + m ≤ N && 1 ≤ n - m ≤ N) || continue
                    r = z[n + m] * conj(z[n - m])
                else
                    g = G[im]
                    r = zero(Complex{T})
                    for (iμ, μ) in enumerate(-U:U)
                        a, b = n + μ + m, n + μ - m
                        (1 ≤ a ≤ N && 1 ≤ b ≤ N) || continue
                        r += g[iμ] * z[a] * conj(z[b])
                    end
                end
                buf[mod(m, nfft) + 1] = h[im] * r
            end
            plan * buf
            @inbounds for k in 1:nfft
                D[k, j] = real(buf[k])
            end
        end
    end
    freq = T[(k - 1) * sr / (2 * nfft) for k in 1:nfft]
    info = CohenSetup(sr, get_winsize(frames), get_step(frames), get_offset(frames), kernel, Float64(sigma), smooth)
    return CohenDistribution{T}(max.(D, zero(T)), D, freq, frames, info)
end

"""
    Wvd(frames::Frames) -> CohenDistribution

Pseudo Wigner-Ville distribution at the centre of every frame: the lag
product `z[n+m] conj(z[n-m])` of the analytic signal, weighted by the
frame window over the lag and Fourier transformed. With a frame length
`L` it has `L` bins at `(k-1)·sr/(2L)`, twice the frequency resolution of
an STFT with the same window, and interference terms between components.

```julia
w = Wvd(Frames(audio; winsize=256, winstep=128, type=hanning))
```
"""
Wvd(frames::Frames) = _cohen(frames, :wvd, 1.0, 0)

"""
    Cwd(frames::Frames; sigma=1, smooth=winsize ÷ 8) -> CohenDistribution

Choi-Williams distribution at the centre of every frame: the lag products
are first smoothed over time with the exponential kernel
`exp(-σ μ² / 4m²)` (normalised, `smooth` samples on each side), which
attenuates the interference terms of the Wigner-Ville distribution; smaller
`sigma` smooths more. Same grid as [`Wvd`](@ref).
"""
Cwd(frames::Frames; sigma::Real=1, smooth::Int=get_winsize(frames) ÷ 8) = begin
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    smooth ≥ 0 || throw(ArgumentError("smooth must be ≥ 0"))
    _cohen(frames, :cwd, sigma, smooth)
end

for S in (:Wvd, :Cwd)
    @eval begin
        $S(audio::AbstractVecOrMat{<:Real}, sr::Int; winsize::Int=256, winstep::Int=winsize ÷ 2,
           type::Base.Callable=hanning, periodic::Bool=true, center::Bool=false, pad_mode::Symbol=:constant, kwargs...) =
            $S(Frames(audio, sr; winsize, winstep, type, periodic, center, pad_mode); kwargs...)
        $S(a::AudioFile; kwargs...) = $S(get_data(a), get_sr(a); kwargs...)
    end
end
