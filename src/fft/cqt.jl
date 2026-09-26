# ---------------------------------------------------------------------------- #
#                         constant-Q / variable-Q transform                    #
# ---------------------------------------------------------------------------- #
# Spectral-kernel CQT (Brown & Puckette 1992) following audioFlux
# (src/cqt_algorithm.c, src/filterbank/cqt_filterBank.c, MIT licence,
# Copyright (c) 2023 libAudioFlux): per bin a windowed complex exponential of
# length Q·sr/f is centred in an FFT window, transformed, thresholded, and
# applied to the FFT of every analysis frame. audioFlux evaluates the lower
# octaves on a decimated signal; here every bin is evaluated on the full-rate
# signal (the direct method), which is exact but slower.

struct CqtSetup{T<:AbstractFloat} <: AbstractSetup
    sr              :: Int64
    nfft            :: Int64
    winsize         :: Int64
    winstep         :: Int64
    offset          :: Int64
    fmin            :: Float64
    bins_per_octave :: Int64
    nbins           :: Int64
    factor          :: Float64
    gamma           :: Float64
    thresh          :: Float64
    window          :: Base.Callable
    norm            :: Base.Callable
    scale           :: Bool
    spectrum        :: Base.Callable
    lengths         :: Vector{T}
end

"""
    Cqt{T} <: AbstractSpectrogram

Constant-Q (or variable-Q) transform on the time grid of a [`Frames`](@ref)
object: `nbins × frames` power or magnitude coefficients on a geometric
frequency grid of `bins_per_octave` bins per octave starting at `fmin`. It
implements the front-end interface, so `MelSpec`, `Mfcc`, the descriptors
and `Chroma` accept it in place of an `Stft`; [`get_complex`](@ref) gives
the complex coefficients.

Build one with [`Cqt(frames; kwargs...)`](@ref Cqt(::Frames)) or
[`Cqt(audio; kwargs...)`](@ref Cqt(::AudioFile)).
"""
struct Cqt{T<:AbstractFloat} <: AbstractSpectrogram
    spec   :: Matrix{T}
    freq   :: Vector{T}
    frames :: Frames{T}
    info   :: CqtSetup{T}
    cplx   :: Maybe{Matrix{Complex{T}}}
end

Base.eltype(::Cqt{T}) where T = T

"""
    get_data(c::Cqt) -> Matrix

The coefficients as stored, `bins × frames` (same as [`get_spec`](@ref)).
"""
@inline get_data(c::Cqt) = c.spec
@inline get_spec(c::Cqt) = c.spec

"""
    get_freq(c::Cqt) -> Vector

Centre frequency of every bin in Hz, `fmin · 2^(k / bins_per_octave)`.
"""
@inline get_freq(c::Cqt)     = c.freq
@inline get_setup(c::Cqt)    = c.info
@inline get_sr(c::Cqt)       = c.info.sr
@inline get_spectrum(c::Cqt) = c.info.spectrum
@inline get_winsize(c::Cqt)  = c.info.winsize
@inline get_step(c::Cqt)     = c.info.winstep
@inline get_overlap(c::Cqt)  = c.info.winsize - c.info.winstep
@inline get_offset(c::Cqt)   = c.info.offset
@inline get_frames(c::Cqt)   = c.frames
@inline get_parent(c::Cqt)   = c.frames
@inline get_energy(c::Cqt)   = get_energy(c.frames)

"""
    get_nfft(c::Cqt) -> Int

Length of the FFT window of the lowest octave (the longest one; the top
octave uses `nfft / 2^(noctaves-1)`).
"""
@inline get_nfft(c::Cqt) = c.info.nfft

"""
    get_bandwidth(c::Cqt) -> Vector

Kernel length of every bin in samples (`Q · sr / f`, or the variable-Q
length when `gamma > 0`).
"""
@inline get_bandwidth(c::Cqt) = c.info.lengths

function Base.show(io::IO, c::Cqt{T}) where T
    nb, nf = size(c.spec)
    print(io, "Cqt{$T}($nf frames × $nb bins, $(c.info.bins_per_octave)/octave from $(round(c.info.fmin, digits=2)) Hz, sr=$(c.info.sr) Hz)")
end

function Base.show(io::IO, ::MIME"text/plain", c::Cqt{T}) where T
    nb, nf = size(c.spec)
    println(io, "Cqt{$T}")
    println(io, "    Sample rate:     $(c.info.sr) Hz")
    println(io, "    Frames:          $nf")
    println(io, "    Bins:            $nb ($(c.info.bins_per_octave) per octave)")
    println(io, "    Frequency range: $(round(c.freq[1], digits=2)) - $(round(c.freq[end], digits=2)) Hz")
    println(io, "    Kernel FFT:      $(c.info.nfft) samples")
    println(io, "    Q factor:        $(c.info.factor), gamma $(c.info.gamma)")
    println(io, "    Hop size:        $(c.info.winstep) samples")
    print(io,   "    Spectrum type:   $(c.info.spectrum)")
end

# ---------------------------------------------------------------------------- #
#                                   kernels                                    #
# ---------------------------------------------------------------------------- #
# one sparse spectral kernel per bin: the one-sided FFT indices it touches and
# the complex weights there. Bins are grouped by octave counted from the top;
# octave `d` below the top is evaluated with an FFT of `nfft_top · 2^d`
# samples, which at full rate is audioFlux's evaluation of the same kernel with
# the top-octave FFT on a signal decimated `d` times. The kernel spectrum is
# therefore sampled and thresholded exactly like audioFlux's.
struct _CqtKernel{T}
    idx :: Vector{Int}
    val :: Vector{Complex{T}}
end

struct _CqtGroup{T}
    nfft    :: Int
    bins    :: UnitRange{Int}
    kernels :: Vector{_CqtKernel{T}}
end

# FFT length of the top octave and the bin ranges of every octave (top first)
function _cqt_octaves(nbins::Int, bins_per_octave::Int, lengths::AbstractVector)
    groups = UnitRange{Int}[]
    hi = nbins
    while hi ≥ 1
        lo = max(1, hi - bins_per_octave + 1)
        push!(groups, lo:hi)
        hi = lo - 1
    end
    nfft_top = nextpow(2, ceil(Int, lengths[first(groups[1])]))
    return nfft_top, groups
end

function _cqt_kernels(::Type{T}, sr::Int, freq::AbstractVector, lengths::AbstractVector,
                      window::Base.Callable, norm::Base.Callable, thresh::Real,
                      bins_per_octave::Int) where T
    n = length(freq)
    ratio = 2.0^(1 / bins_per_octave)
    nfft_top, octs = _cqt_octaves(n, bins_per_octave, lengths)
    th2 = T(thresh)^2
    groups = _CqtGroup{T}[]
    for (d, bins) in enumerate(octs)
        nfft = nfft_top * 2^(d - 1)
        half = nfft ÷ 2 + 1
        plan = plan_fft(zeros(Complex{T}, nfft))
        kernels = Vector{_CqtKernel{T}}(undef, length(bins))
        for (m, k) in enumerate(bins)
            len = min(ceil(Int, lengths[k]), nfft)
            w = _make_window(T, window, len, true)
            tk = zeros(Complex{T}, nfft)
            start = (nfft - len) ÷ 2
            weight = norm === none_norm ? T(lengths[k]) : one(T)
            f = T(freq[k])
            @inbounds for j in 0:len-1
                tk[start + j + 1] = w[j + 1] * cis(T(2π) * j * f / T(sr)) / weight
            end
            if norm === area
                tk ./= sum(abs, tk)
            elseif norm === bandwidth
                lo = k > 1 ? freq[k - 1] : freq[k] / ratio
                hi = k < n ? freq[k + 1] : freq[k] * ratio
                tk ./= T((hi - lo) / 2)
            end
            tk .*= T(lengths[k]) / T(nfft)
            K = plan * tk
            idx = Int[]; val = Complex{T}[]
            @inbounds for i in 1:half
                if abs2(K[i]) > th2
                    push!(idx, i); push!(val, K[i])
                end
            end
            kernels[m] = _CqtKernel{T}(idx, val)
        end
        push!(groups, _CqtGroup{T}(nfft, bins, kernels))
    end
    return groups
end

# the complex CQT of every frame: for every octave an nfft window centred on
# the frame centre (zero outside the signal), one real FFT, one sparse dot
# product per bin
function _cqt!(out::Matrix{Complex{T}}, frames::Frames{T}, groups::Vector{_CqtGroup{T}},
               scale::Bool, lengths::Vector{T}) where T
    x  = get_signal(frames)
    N  = length(x)
    n  = length(frames)
    n == 0 && return out
    winsize = get_winsize(frames)
    plans = [plan_rfft(zeros(T, g.nfft)) for g in groups]
    sq = scale ? T[sqrt(l) for l in lengths] : T[]
    Threads.@threads for chunk in _chunks(n)
        bufs = [zeros(T, g.nfft) for g in groups]
        Xs   = [Vector{Complex{T}}(undef, g.nfft ÷ 2 + 1) for g in groups]
        for j in chunk
            c = frames.starts[j] + winsize ÷ 2            # centre, 1-based
            for (gi, g) in enumerate(groups)
                buf, X = bufs[gi], Xs[gi]
                lo = c - g.nfft ÷ 2
                fill!(buf, zero(T))
                @inbounds for i in max(1, 1 - lo + 1):min(g.nfft, N - lo + 1)
                    buf[i] = x[lo + i - 1]
                end
                mul!(X, plans[gi], buf)
                @inbounds for (m, k) in enumerate(g.bins)
                    ker = g.kernels[m]
                    acc = zero(Complex{T})
                    for (q, i) in enumerate(ker.idx)
                        acc += X[i] * ker.val[q]
                    end
                    out[k, j] = scale ? acc / sq[k] : acc
                end
            end
        end
    end
    return out
end

"""
    Cqt(frames::Frames; kwargs...) -> Cqt

Constant-Q transform of the signal of `frames`, one column per frame.

For every bin `k` a kernel `w(n) exp(2πi n f_k / sr)` of length
`Q · sr / (f_k + γ / (2^(1/b) − 1))` (with `Q = factor / (2^(1/b) − 1)`,
`b = bins_per_octave`) is centred in an FFT window, transformed, scaled by
`length / nfft` and kept where its magnitude exceeds `thresh`. The top
octave uses the next power of two above its longest kernel and every octave
below doubles it (audioFlux's decimation scheme, evaluated at full rate).
Every frame is analysed with those windows centred on the frame centre
(zero-padded at the signal ends), so the `Frames` object only fixes the
time grid: `winsize` is the grid shared with `Stft`, not the analysis
length. `γ = 0` is the CQT, `γ > 0` the
variable-Q transform of Schörkhuber et al. (audioFlux's `beta`).

# Keyword Arguments
- `fmin::Real=32.703` (C1): frequency of the first bin
- `bins_per_octave::Int=12`
- `nbins::Int=84`: number of bins (multiple of `bins_per_octave` to cover whole octaves)
- `factor::Real=1`: scales Q (`< 1` shortens the kernels)
- `gamma::Real=0`: variable-Q bandwidth offset in Hz
- `thresh::Real=0.01`: kernel sparsity threshold on the scaled kernel
  spectrum (`0` keeps every bin)
- `window::Base.Callable=hanning`: kernel window (periodic)
- `norm::Base.Callable=area`: kernel normalisation, `area` (unit L1 norm,
  audioFlux's Python default), `none_norm` (divide by the kernel length) or
  `bandwidth`
- `scale::Bool=true`: divide every bin by the square root of its kernel length
- `spectrum::Base.Callable=power`: `power` or `magnitude`
- `keep_complex::Bool=false`: keep the complex coefficients for [`get_complex`](@ref)

# Examples
```julia
frames = Frames(audio; winsize=512, winstep=256, center=true)
cqt    = Cqt(frames; fmin=note_to_hz("C1"), bins_per_octave=24, nbins=168)
chroma = Chroma(cqt)
mel    = MelSpec(cqt; nbands=26)     # any downstream stage
vqt    = Cqt(frames; gamma=20)
```

See also [`Stft`](@ref), [`Cwt`](@ref), [`cqt_frequencies`](@ref).
"""
function Cqt(
    frames          :: Frames{T};
    fmin            :: Real=32.703,
    bins_per_octave :: Int64=12,
    nbins           :: Int64=84,
    factor          :: Real=1,
    gamma           :: Real=0,
    thresh          :: Real=0.01,
    window          :: Base.Callable=hanning,
    norm            :: Base.Callable=area,
    scale           :: Bool=true,
    spectrum        :: Base.Callable=power,
    keep_complex    :: Bool=false,
) where {T<:AbstractFloat}
    sr = get_sr(frames)
    fmin > 0 || throw(ArgumentError("fmin must be positive, got $fmin"))
    bins_per_octave ≥ 1 || throw(ArgumentError("bins_per_octave must be ≥ 1"))
    nbins ≥ 1 || throw(ArgumentError("nbins must be ≥ 1"))
    factor > 0 || throw(ArgumentError("factor must be positive"))
    gamma ≥ 0 || throw(ArgumentError("gamma must be ≥ 0"))
    thresh ≥ 0 || throw(ArgumentError("thresh must be ≥ 0"))
    norm in (area, none_norm, bandwidth) || throw(ArgumentError(
        "norm must be `area`, `none_norm` or `bandwidth`, got $norm"))
    spectrum in (power, magnitude) ||
        throw(ArgumentError("spectrum must be `power` or `magnitude`, got $spectrum"))

    freq = T[fmin * 2.0^((k - 1) / bins_per_octave) for k in 1:nbins]
    freq[end] < sr / 2 || throw(ArgumentError(
        "the highest bin ($(round(freq[end], digits=1)) Hz) reaches the Nyquist frequency; reduce nbins or fmin"))
    r = 2.0^(1 / bins_per_octave) - 1
    Q = factor / r
    lengths = T[Q * sr / (f + gamma / r) for f in freq]
    groups = _cqt_kernels(T, sr, freq, lengths, window, norm, thresh, bins_per_octave)
    nfft = groups[end].nfft

    C = Matrix{Complex{T}}(undef, nbins, length(frames))
    _cqt!(C, frames, groups, scale, lengths)
    spec = spectrum(C)

    info = CqtSetup{T}(sr, nfft, get_winsize(frames), get_step(frames), get_offset(frames),
                       Float64(fmin), bins_per_octave, nbins, Float64(factor), Float64(gamma),
                       Float64(thresh), window, norm, scale, spectrum, lengths)
    return Cqt{T}(spec, freq, frames, info, keep_complex ? C : nothing)
end

"""
    Cqt(audio::AudioFile; kwargs...) -> Cqt
    Cqt(x::AbstractVecOrMat, sr::Int; kwargs...) -> Cqt

Frame the signal and compute its constant-Q transform. Framing keywords
(`winsize`, `winstep`, `center`, ...) go to [`Frames`](@ref), the rest to
[`Cqt(::Frames)`](@ref).
"""
function Cqt(
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
    return Cqt(frames; kwargs...)
end

Cqt(a::AudioFile; kwargs...) = Cqt(get_data(a), get_sr(a); kwargs...)

"""
    get_complex(c::Cqt) -> Matrix{Complex}

The complex constant-Q coefficients, `bins × frames`: the matrix kept with
`keep_complex=true`, otherwise recomputed from the frames.
"""
function get_complex(c::Cqt{T}) where T
    isnothing(c.cplx) || return c.cplx
    i = c.info
    groups = _cqt_kernels(T, i.sr, c.freq, i.lengths, i.window, i.norm, i.thresh, i.bins_per_octave)
    C = Matrix{Complex{T}}(undef, i.nbins, length(c.frames))
    return _cqt!(C, c.frames, groups, i.scale, i.lengths)
end
