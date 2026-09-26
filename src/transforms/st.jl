# ---------------------------------------------------------------------------- #
#                        S-transform and fast S-transform                      #
# ---------------------------------------------------------------------------- #
# Ports of audioFlux's ST and FST (src/st_algorithm.c, src/fst_algorithm.c,
# MIT licence, Copyright (c) 2023 libAudioFlux).

"""
    st(x; min_index=1, max_index=length(x)÷2, factor=1, norm=1) -> Matrix{Complex}

Stockwell transform of a whole signal (audioFlux `ST`). For every frequency
index `k` in `min_index:max_index` the row is
`ifft(X[k + f] · G_k[f])` with the Gaussian voice
`G_k[f] = exp(-2π² λ f² / k^(2p)) + exp(-2π² λ (f - N)² / k^(2p))`
(`λ = factor`, `p = norm`); the row for `k = 0` is the signal mean. Returns
`(max_index - min_index + 1) × length(x)` complex values; row `i` is the
frequency `(min_index + i - 1) · sr / N`.
"""
function st(x::AbstractVector{<:Real}; min_index::Int=1, max_index::Int=length(x) ÷ 2,
            factor::Real=1, norm::Real=1)
    T = float(eltype(x))
    N = length(x)
    (0 ≤ min_index ≤ max_index ≤ N ÷ 2) || throw(ArgumentError(
        "need 0 ≤ min_index ≤ max_index ≤ N/2, got $min_index:$max_index with N = $N"))
    factor > 0 || throw(ArgumentError("factor must be positive"))
    norm > 0 || throw(ArgumentError("norm must be positive"))
    X = fft(Vector{Complex{T}}(x))
    nb = max_index - min_index + 1
    out = Matrix{Complex{T}}(undef, nb, N)
    plan = plan_ifft!(zeros(Complex{T}, N))
    f1 = T[(j)^2 for j in 0:N-1]
    f2 = T[(j - N)^2 for j in 0:N-1]
    Threads.@threads for chunk in _chunks(nb)
        buf = zeros(Complex{T}, N)
        for i in chunk
            k = min_index + i - 1
            if k == 0
                out[i, :] .= Complex{T}(sum(x) / N)
                continue
            end
            c = -T(factor) * T(2π^2) / T(k)^(2 * T(norm))
            @inbounds for f in 1:N
                g = exp(c * f1[f]) + exp(c * f2[f])
                buf[f] = X[mod1(k + f, N)] * g
            end
            plan * buf
            @inbounds out[i, :] .= buf
        end
    end
    return out
end

# dyadic partition lengths of the fast S-transform for N = 2^m
function _fst_partition(m::Int)
    L = 2m
    len = ones(Int, L)
    for (i, j) in zip(1:m-2, m-2:-1:1)
        len[i + 1] = 2^j
    end
    for i in m+2:L
        len[i] = 2^(i - m - 2)
    end
    return len
end

# the dyadic blocks of the fast S-transform of a 2^m signal: the transformed
# spectrum `Y` (block by block), block lengths, the first value index and the
# first image row (0-based) of every block
struct _FstBlocks{T}
    Y        :: Vector{Complex{T}}
    len      :: Vector{Int}
    vstart   :: Vector{Int}
    rowstart :: Vector{Int}
    N        :: Int
end

function _fst_blocks(x::AbstractVector{<:Real})
    T = float(eltype(x))
    N = length(x)
    ispow2(N) && N ≥ 8 || throw(ArgumentError("fst needs a power-of-two length ≥ 8, got $N"))
    m = round(Int, log2(N))
    # fft of the ifftshifted signal, then fftshift, orthonormal scaling
    xs = vcat(x[N÷2+1:end], x[1:N÷2])
    X  = fft(Vector{Complex{T}}(xs))
    X  = vcat(X[N÷2+1:end], X[1:N÷2]) ./ T(sqrt(N))
    len = _fst_partition(m)
    Y = copy(X)
    vstart = similar(len); rowstart = similar(len)
    idx = 1; cum = 0
    for i in 1:2m
        l = len[i]
        if l > 1
            blk = X[idx:idx+l-1]
            blk = vcat(blk[l÷2+1:end], blk[1:l÷2])          # ifftshift
            y = ifft(blk) .* T(sqrt(l))
            Y[idx:idx+l-1] .= vcat(y[l÷2+1:end], y[1:l÷2])   # fftshift
        end
        vstart[i] = idx
        cum += l
        rowstart[i] = N - cum
        idx += l
    end
    return _FstBlocks{T}(Y, len, vstart, rowstart, N)
end

# the image row of frequency index k (row N/2 - k, 0-based): the samples of
# its block, each held for N/len columns
function _fst_row!(buf::AbstractVector{Complex{T}}, b::_FstBlocks{T}, k::Int) where T
    N = b.N
    r = N ÷ 2 - k
    i = findfirst(j -> b.rowstart[j] ≤ r < b.rowstart[j] + b.len[j], eachindex(b.len))
    isnothing(i) && throw(ArgumentError("frequency index $k is outside 0:$(N ÷ 2)"))
    l = b.len[i]; hold = N ÷ l
    @inbounds for j in 0:l-1
        v = b.Y[b.vstart[i] + j]
        for c in hold*j+1:hold*(j+1)
            buf[c] = v
        end
    end
    return buf
end

"""
    fst(x; min_index=1, max_index=length(x)÷2) -> Matrix{Complex}

Fast S-transform of a whole signal of length `2^m` (Brown, Lauzon & Frayne
2010; audioFlux `FST`): the shifted spectrum is cut into dyadic blocks
(`1, 2^(m-2), …, 2, 1, 1, 1, 2, …, 2^(m-2)`), every block is inverse
transformed at its own length (orthonormal scaling) and written to the
`(N/2+1) × N` image with every block sample held for `N / len` columns.
Rows `min_index:max_index` of that image are returned, in audioFlux's
row order (row `i` is frequency index `min_index + i - 1`).
"""
function fst(x::AbstractVector{<:Real}; min_index::Int=1, max_index::Int=length(x) ÷ 2)
    N = length(x)
    (0 ≤ min_index ≤ max_index ≤ N ÷ 2) || throw(ArgumentError(
        "need 0 ≤ min_index ≤ max_index ≤ N/2, got $min_index:$max_index"))
    b = _fst_blocks(x)
    T = float(eltype(x))
    out = Matrix{Complex{T}}(undef, max_index - min_index + 1, N)
    buf = Vector{Complex{T}}(undef, N)
    for (i, k) in enumerate(min_index:max_index)
        _fst_row!(buf, b, k)
        out[i, :] .= buf
    end
    return out
end

# ---------------------------------------------------------------------------- #
#                                  front ends                                  #
# ---------------------------------------------------------------------------- #
struct StSetup{T<:AbstractFloat} <: AbstractSetup
    sr        :: Int64
    winsize   :: Int64
    winstep   :: Int64
    offset    :: Int64
    spectrum  :: Base.Callable
    min_index :: Int64
    max_index :: Int64
    factor    :: Float64
    norm      :: Float64
    fast      :: Bool
end

"""
    St{T} <: AbstractSpectrogram

S-transform of a signal pooled on the time grid of a [`Frames`](@ref)
object: `bins × frames`, power or magnitude, on the FFT grid
`min_index:max_index` of the whole signal. Implements the front-end
interface. See [`St(frames; kwargs...)`](@ref St(::Frames)).
"""
struct St{T<:AbstractFloat} <: AbstractSpectrogram
    spec   :: Matrix{T}
    freq   :: Vector{T}
    frames :: Frames{T}
    info   :: StSetup{T}
end
@pooled_frontend St
Base.show(io::IO, s::St{T}) where T =
    print(io, "St{$T}($(size(s.spec, 2)) frames × $(size(s.spec, 1)) bins, $(s.info.fast ? "fast" : "gaussian"), sr=$(s.info.sr) Hz)")

"""
    Fst{T} <: AbstractSpectrogram

Fast S-transform pooled on the grid of a [`Frames`](@ref) object; same
layout as [`St`](@ref). See [`Fst(frames; kwargs...)`](@ref Fst(::Frames)).
"""
struct Fst{T<:AbstractFloat} <: AbstractSpectrogram
    spec   :: Matrix{T}
    freq   :: Vector{T}
    frames :: Frames{T}
    info   :: StSetup{T}
end
@pooled_frontend Fst
Base.show(io::IO, s::Fst{T}) where T =
    print(io, "Fst{$T}($(size(s.spec, 2)) frames × $(size(s.spec, 1)) bins, sr=$(s.info.sr) Hz)")

# the whole signal is zero-padded to a power of two for the fast transform
function _pow2_signal(x::Vector{T}) where T
    N = length(x)
    M = max(8, nextpow(2, N))
    M == N && return x
    return vcat(x, zeros(T, M - N))
end

"""
    St(frames::Frames; freqrange=(0, sr÷2), factor=1, norm=1, spectrum=power) -> St

Compute [`st`](@ref) on the signal of `frames` for the FFT bins inside
`freqrange` and pool every bin over the frames (window weighted). Memory is
one signal-length buffer per thread plus the output: the rows are pooled as
they are produced.
"""
function St(frames::Frames{T}; freqrange::FreqRange=(0, get_sr(frames) ÷ 2), factor::Real=1,
            norm::Real=1, spectrum::Base.Callable=power) where T
    _check_spectrum(spectrum)
    factor > 0 || throw(ArgumentError("factor must be positive"))
    norm > 0 || throw(ArgumentError("norm must be positive"))
    sr = get_sr(frames)
    x  = get_signal(frames)
    N  = length(x)
    grid = (0:N÷2) .* (T(sr) / T(N))
    idx  = _freq_indices(grid, freqrange)
    kmin, kmax = first(idx) - 1, last(idx) - 1
    nb = kmax - kmin + 1
    spec = Matrix{T}(undef, nb, length(frames))
    # row by row to bound memory
    X = fft(Vector{Complex{T}}(x))
    f1 = T[(j)^2 for j in 0:N-1]
    f2 = T[(j - N)^2 for j in 0:N-1]
    plan = plan_ifft!(zeros(Complex{T}, N))
    Threads.@threads for chunk in _chunks(nb)
        buf = zeros(Complex{T}, N)
        for i in chunk
            k = kmin + i - 1
            if k == 0
                fill!(buf, Complex{T}(sum(x) / N))
            else
                c = -T(factor) * T(2π^2) / T(k)^(2 * T(norm))
                @inbounds for f in 1:N
                    buf[f] = X[mod1(k + f, N)] * (exp(c * f1[f]) + exp(c * f2[f]))
                end
                plan * buf
            end
            _pool_band!(spec, i, buf, frames, spectrum)
        end
    end
    info = StSetup{T}(sr, get_winsize(frames), get_step(frames), get_offset(frames), spectrum,
                      kmin, kmax, Float64(factor), Float64(norm), false)
    return St{T}(spec, Vector{T}(grid[idx]), frames, info)
end

"""
    Fst(frames::Frames; freqrange=(0, sr÷2), spectrum=power) -> Fst

Fast S-transform ([`fst`](@ref)) of the signal of `frames`, zero-padded to
a power of two, pooled over the frames for the bins inside `freqrange`.
"""
function Fst(frames::Frames{T}; freqrange::FreqRange=(0, get_sr(frames) ÷ 2), spectrum::Base.Callable=power) where T
    _check_spectrum(spectrum)
    sr = get_sr(frames)
    xp = _pow2_signal(get_signal(frames))
    N  = length(xp)
    grid = (0:N÷2) .* (T(sr) / T(N))
    idx  = _freq_indices(grid, freqrange)
    kmin, kmax = first(idx) - 1, last(idx) - 1
    b = _fst_blocks(xp)
    spec = Matrix{T}(undef, kmax - kmin + 1, length(frames))
    buf = Vector{Complex{T}}(undef, N)
    for (i, k) in enumerate(kmin:kmax)
        _fst_row!(buf, b, k)
        _pool_band!(spec, i, buf, frames, spectrum)
    end
    info = StSetup{T}(sr, get_winsize(frames), get_step(frames), get_offset(frames), spectrum,
                      kmin, kmax, 1.0, 1.0, true)
    return Fst{T}(spec, Vector{T}(grid[idx]), frames, info)
end

for S in (:St, :Fst)
    @eval begin
        $S(audio::AbstractVecOrMat{<:Real}, sr::Int; winsize::Int=sr ≤ 8000 ? 256 : 512, winstep::Int=winsize ÷ 2,
           type::Base.Callable=rect, periodic::Bool=true, center::Bool=false, pad_mode::Symbol=:constant, kwargs...) =
            $S(Frames(audio, sr; winsize, winstep, type, periodic, center, pad_mode); kwargs...)
        $S(a::AudioFile; kwargs...) = $S(get_data(a), get_sr(a); kwargs...)
    end
end

"""
    get_complex(s::St), get_complex(s::Fst) -> Matrix{Complex}

The complex transform sampled at the frame centres, `bins × frames`
(recomputed with [`st`](@ref) or [`fst`](@ref)).
"""
function get_complex(s::St{T}) where T
    i = s.info
    Y = st(get_signal(s.frames); min_index=i.min_index, max_index=i.max_index, factor=i.factor, norm=i.norm)
    return Y[:, _frame_centres(s.frames)]
end
function get_complex(s::Fst{T}) where T
    i = s.info
    b = _fst_blocks(_pow2_signal(get_signal(s.frames)))
    c = _frame_centres(s.frames)
    out = Matrix{Complex{T}}(undef, i.max_index - i.min_index + 1, length(c))
    buf = Vector{Complex{T}}(undef, b.N)
    for (r, k) in enumerate(i.min_index:i.max_index)
        _fst_row!(buf, b, k)
        out[r, :] .= buf[c]
    end
    return out
end
