# ---------------------------------------------------------------------------- #
#               discrete, wave-packet and stationary wavelet transforms        #
# ---------------------------------------------------------------------------- #
# Ports of audioFlux's DWT, WPT and SWT (src/dwt_algorithm.c,
# src/wpt_algorithm.c, src/swt_algorithm.c, MIT licence, Copyright (c) 2023
# libAudioFlux): Mallat's cascade with periodic extension, the full
# wave-packet tree in frequency order, and the undecimated (à trous)
# transform. The filter tables are in dwt_coefs.jl.

"""
    DISCRETE_WAVELETS

Names of the 51 discrete wavelets accepted by [`wavelet_filters`](@ref),
[`dwt`](@ref), [`wpt`](@ref) and [`swt`](@ref) (audioFlux's families):
`"haar"`, Daubechies `"db2"`–`"db10"`, `"db20"`, `"db30"`, `"db40"`,
symlets `"sym2"`–`"sym10"`, `"sym20"`, `"sym30"`, coiflets `"coif1"`–`"coif5"`,
Fejér-Korovkin `"fk4"`, `"fk6"`, `"fk8"`, `"fk14"`, `"fk18"`, `"fk22"`,
biorthogonal `"bior1.1"`–`"bior6.8"` and the discrete Meyer `"dmey"`.
"""
DISCRETE_WAVELETS

"""
    wavelet_filters(name) -> (loD, hiD, loR, hiR)

Decomposition and reconstruction filters of a discrete wavelet: `"haar"`,
`"db2"`–`"db10"`, `"db20"`, `"db30"`, `"db40"`, `"sym2"`–`"sym10"`,
`"sym20"`, `"sym30"`, `"coif1"`–`"coif5"`, `"fk4"`, `"fk6"`, `"fk8"`,
`"fk14"`, `"fk18"`, `"fk22"`, `"bior1.1"`, `"bior1.3"`, `"bior1.5"`,
`"bior2.2"`–`"bior2.8"`, `"bior3.1"`–`"bior3.9"`, `"bior4.4"`, `"bior5.5"`,
`"bior6.8"`, `"dmey"` (audioFlux's tables, six decimals). `"db1"` is `"haar"`.
"""
function wavelet_filters(name::AbstractString)
    key = name == "db1" ? "haar" : String(name)
    haskey(_DWT_FILTERS, key) || throw(ArgumentError(
        "unknown discrete wavelet \"$name\"; available: $(join(DISCRETE_WAVELETS, ", "))"))
    return _DWT_FILTERS[key]
end

# periodic extension by half the filter length on both sides (audioFlux
# __periodPadding): the last h samples, the signal, the first h samples,
# wrapping around when the signal is shorter than h
function _period_pad(a::AbstractVector{T}, L::Int) where T
    n = length(a)
    h = L ÷ 2
    return T[a[mod1(i, n)] for i in (1 - h):(n + L - h)]
end

# linear convolution y = a ∗ f, `full` (length n+L-1) or `valid` (n-L+1)
function _conv(a::AbstractVector{T}, f::AbstractVector, mode::Symbol) where T
    n, L = length(a), length(f)
    full = zeros(T, n + L - 1)
    @inbounds for j in 1:L
        fj = T(f[j])
        iszero(fj) && continue
        @simd for i in 1:n
            full[i + j - 1] += a[i] * fj
        end
    end
    return mode === :full ? full : full[L:n]
end

# one analysis step: periodic padding, valid convolution, keep the odd samples
function _dwt_step(a::AbstractVector{T}, loD, hiD) where T
    L  = length(loD)
    p  = _period_pad(a, L)
    ca = _conv(p, loD, :valid)
    cd = _conv(p, hiD, :valid)
    n2 = length(a) ÷ 2
    return T[ca[2i] for i in 1:n2], T[cd[2i] for i in 1:n2]
end

function _check_levels(N::Int, level::Int)
    level ≥ 1 || throw(ArgumentError("level must be ≥ 1, got $level"))
    N % (1 << level) == 0 || throw(ArgumentError(
        "the signal length ($N) must be a multiple of 2^level = $(1 << level)"))
end

"""
    dwt(x; wavelet="sym4", level=log2(length(x)) - 1) -> (coefs, image)

Discrete wavelet transform of a whole signal (Mallat's algorithm with
periodic extension; audioFlux `DWT`). The length must be a multiple of
`2^level`. `coefs` is `[cA_level; cD_level; …; cD_1]` (the approximation of
the last level first, then the details from the coarsest to the finest,
PyWavelets' `wavedec` order, `length(x)` values in all). `image` is
audioFlux's `level × length(x)` matrix: row `i` holds `coefs[2^i+1 : 2^(i+1)]`,
every coefficient repeated `length(x) / 2^i` times (with the full depth
`level = log2(N) - 1`, row `i` is the detail band of level `log2(N) - i`).
"""
function dwt(x::AbstractVector{<:Real}; wavelet::AbstractString="sym4",
             level::Int=round(Int, log2(length(x))) - 1)
    T = float(eltype(x))
    N = length(x)
    _check_levels(N, level)
    loD, hiD, _, _ = wavelet_filters(wavelet)
    a = Vector{T}(x)
    coefs = Vector{T}(undef, N)
    hi = N
    for _ in 1:level
        a, d = _dwt_step(a, loD, hiD)
        coefs[hi - length(d) + 1:hi] .= d
        hi -= length(d)
    end
    coefs[1:hi] .= a
    return coefs, _dwt_image(coefs, level)
end

function _dwt_image(coefs::AbstractVector{T}, level::Int) where T
    N = length(coefs)
    img = zeros(T, level, N)
    for i in 1:level
        s = 1 << i
        s > N && break
        blen = min(s, N - s)
        hold = N ÷ s
        for j in 0:N-1
            l = j ÷ hold
            l < blen && (img[i, j + 1] = coefs[s + l + 1])
        end
    end
    return img
end

"""
    wpt(x; wavelet="sym4", level=log2(length(x)) - 1) -> (coefs, image)

Wave-packet transform of a whole signal (audioFlux `WPT`): the full binary
tree of `level` analysis steps, the children of every high-pass node
swapped so that the `2^level` leaves are in frequency order. `coefs` holds
the leaves one after the other (`length(x)` values), `image` is
`2^level × length(x)` with leaf `k` (band `[k-1, k] · sr / 2^(level+1)`)
repeated over the samples it covers.
"""
function wpt(x::AbstractVector{<:Real}; wavelet::AbstractString="sym4",
             level::Int=round(Int, log2(length(x))) - 1)
    T = float(eltype(x))
    N = length(x)
    _check_levels(N, level)
    loD, hiD, _, _ = wavelet_filters(wavelet)
    nodes = Vector{Vector{T}}(undef, (1 << (level + 1)) - 1)   # heap order
    nodes[1] = Vector{T}(x)
    for i in 0:(1 << level) - 2
        a, d = _dwt_step(nodes[i + 1], loD, hiD)
        swap = i > 0 && iseven(i)             # a right (high-pass) child
        nodes[2i + 2] = swap ? d : a
        nodes[2i + 3] = swap ? a : d
    end
    leaves = nodes[(1 << level):end]
    coefs = reduce(vcat, leaves)
    len = N >> level
    img = Matrix{T}(undef, length(leaves), N)
    for (k, leaf) in enumerate(leaves), j in 0:N-1
        img[k, j + 1] = leaf[j ÷ (N ÷ len) + 1]
    end
    return coefs, img
end

"""
    swt(x; wavelet="sym4", level=1) -> (A, D)

Stationary (undecimated, à trous) wavelet transform of a whole signal
(audioFlux `SWT`): at level `i` the approximation of level `i-1` is
periodically extended, convolved with the decomposition filters upsampled
by `2^(i-1)` and cut back to `length(x)` samples. `A` and `D` are
`level × length(x)`: the approximation and the detail of every level. The
length must be a multiple of `2^level`.
"""
function swt(x::AbstractVector{<:Real}; wavelet::AbstractString="sym4", level::Int=1)
    T = float(eltype(x))
    N = length(x)
    _check_levels(N, level)
    loD, hiD, _, _ = wavelet_filters(wavelet)
    L = length(loD)
    A = Matrix{T}(undef, level, N)
    D = Matrix{T}(undef, level, N)
    a  = Vector{T}(x)
    lo = Vector{T}(loD); hi = Vector{T}(hiD)
    up = L
    for i in 1:level
        p  = _period_pad(a, up)
        ca = _conv(p, lo, :full)
        cd = _conv(p, hi, :full)
        A[i, :] .= view(ca, up+1:up+N)
        D[i, :] .= view(cd, up+1:up+N)
        a = A[i, :]
        # filters upsampled by 2^i, padded to twice the previous length
        lo = zeros(T, 2up); hi = zeros(T, 2up)
        for j in 1:L
            lo[(j - 1) * (1 << i) + 1] = loD[j]
            hi[(j - 1) * (1 << i) + 1] = hiD[j]
        end
        up *= 2
    end
    return A, D
end

# ---------------------------------------------------------------------------- #
#                                  front ends                                  #
# ---------------------------------------------------------------------------- #
struct DiscreteWaveletSetup <: AbstractSetup
    sr       :: Int64
    winsize  :: Int64
    winstep  :: Int64
    offset   :: Int64
    spectrum :: Base.Callable
    method   :: Symbol          # :dwt, :wpt or :swt
    wavelet  :: String
    level    :: Int64
end

"""
    DiscreteWavelet{T} <: AbstractSpectrogram

A discrete wavelet decomposition pooled on the time grid of a
[`Frames`](@ref) object, one row per subband in ascending frequency: built
by [`Dwt`](@ref) (approximation plus one detail band per level), [`Wpt`](@ref)
(`2^level` equal bands) or [`Swt`](@ref) (like `Dwt`, undecimated). Every
coefficient is held over the samples it covers and its power (or
magnitude) is averaged over each frame, weighted by the frame window. It
implements the front-end interface.
"""
struct DiscreteWavelet{T<:AbstractFloat} <: AbstractSpectrogram
    spec   :: Matrix{T}
    freq   :: Vector{T}
    frames :: Frames{T}
    info   :: DiscreteWaveletSetup
end
@pooled_frontend DiscreteWavelet
get_nbands(d::DiscreteWavelet) = size(d.spec, 1)
Base.show(io::IO, d::DiscreteWavelet{T}) where T =
    print(io, "DiscreteWavelet{$T}(:$(d.info.method), $(d.info.wavelet), $(size(d.spec, 2)) frames × $(size(d.spec, 1)) bands)")

# the signal zero-padded to a multiple of 2^level
function _dyadic_signal(x::Vector{T}, level::Int) where T
    m = 1 << level
    n = cld(length(x), m) * m
    return n == length(x) ? x : vcat(x, zeros(T, n - length(x)))
end

function _pool_rows(rows::AbstractMatrix, frames::Frames{T}, spectrum) where T
    spec = Matrix{T}(undef, size(rows, 1), length(frames))
    for k in axes(rows, 1)
        _pool_band!(spec, k, view(rows, k, :), frames, spectrum)
    end
    return spec
end

# hold every coefficient of a band of `len` values over the `N / len` samples it covers
_hold(c::AbstractVector{T}, N::Int) where T = T[c[(j * length(c)) ÷ N + 1] for j in 0:N-1]

function _discrete_frontend(frames::Frames{T}, method::Symbol, wavelet, level, spectrum) where T
    _check_spectrum(spectrum)
    sr = get_sr(frames)
    x  = _dyadic_signal(get_signal(frames), level)
    N  = length(x)
    if method === :wpt
        _, img = wpt(x; wavelet, level)
        rows = img
        bw   = sr / 2 / (1 << level)
        freq = T[(k - 0.5) * bw for k in 1:(1 << level)]
    else
        # approximation, then the details from the coarsest level to the finest
        if method === :dwt
            coefs, _ = dwt(x; wavelet, level)
            rows = Matrix{T}(undef, level + 1, N)
            rows[1, :] .= _hold(view(coefs, 1:N >> level), N)
            lo = N >> level
            for (r, lev) in enumerate(level:-1:1)
                len = N >> lev
                rows[r + 1, :] .= _hold(view(coefs, lo + 1:lo + len), N)
                lo += len
            end
        else
            A, D = swt(x; wavelet, level)
            rows = vcat(A[level:level, :], D[level:-1:1, :])
        end
        # band of the approximation [0, sr/2^(level+1)], detail j [sr/2^(j+1), sr/2^j]
        freq = vcat(T(sr / 2^(level + 2)), T[0.75 * sr / 2^j for j in level:-1:1])
    end
    info = DiscreteWaveletSetup(sr, get_winsize(frames), get_step(frames), get_offset(frames),
                                spectrum, method, String(wavelet), level)
    return DiscreteWavelet{T}(_pool_rows(rows, frames, spectrum), freq, frames, info)
end

"""
    Dwt(frames::Frames; wavelet="sym4", level=5, spectrum=power) -> DiscreteWavelet

Discrete wavelet transform ([`dwt`](@ref)) of the signal of `frames`
(zero-padded to a multiple of `2^level`), pooled on its frames: row 1 is
the approximation band `[0, sr/2^(level+1)]`, rows `2:level+1` the detail
bands from the coarsest to the finest, `get_freq` their centres.
"""
Dwt(frames::Frames; wavelet::AbstractString="sym4", level::Int=5, spectrum::Base.Callable=power) =
    _discrete_frontend(frames, :dwt, wavelet, level, spectrum)

"""
    Wpt(frames::Frames; wavelet="sym4", level=5, spectrum=power) -> DiscreteWavelet

Wave-packet transform ([`wpt`](@ref)) pooled on the frames: `2^level` bands
of width `sr / 2^(level+1)` in frequency order.
"""
Wpt(frames::Frames; wavelet::AbstractString="sym4", level::Int=5, spectrum::Base.Callable=power) =
    _discrete_frontend(frames, :wpt, wavelet, level, spectrum)

"""
    Swt(frames::Frames; wavelet="sym4", level=5, spectrum=power) -> DiscreteWavelet

Stationary wavelet transform ([`swt`](@ref)) pooled on the frames: the
approximation of the last level and the details from the coarsest level to
the finest, on the bands of [`Dwt`](@ref) but without decimation.
"""
Swt(frames::Frames; wavelet::AbstractString="sym4", level::Int=5, spectrum::Base.Callable=power) =
    _discrete_frontend(frames, :swt, wavelet, level, spectrum)

for S in (:Dwt, :Wpt, :Swt)
    @eval begin
        $S(audio::AbstractVecOrMat{<:Real}, sr::Int; winsize::Int=sr ≤ 8000 ? 256 : 512, winstep::Int=winsize ÷ 2,
           type::Base.Callable=rect, periodic::Bool=true, center::Bool=false, pad_mode::Symbol=:constant, kwargs...) =
            $S(Frames(audio, sr; winsize, winstep, type, periodic, center, pad_mode); kwargs...)
        $S(a::AudioFile; kwargs...) = $S(get_data(a), get_sr(a); kwargs...)
    end
end
