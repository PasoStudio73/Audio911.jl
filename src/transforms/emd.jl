# ---------------------------------------------------------------------------- #
#     adaptive decompositions: EMD, Hilbert-Huang spectrum, empirical wavelets #
# ---------------------------------------------------------------------------- #
# audioFlux declares EMD, EWT and HHT in include/ without an implementation;
# these follow Huang et al. (1998) with Rilling, Flandrin & Gonçalves (2003)
# boundary mirroring, and Gilles (2013).

# natural cubic spline through (xs, ys) evaluated at 1:N (xs strictly increasing)
function _spline!(out::AbstractVector{T}, xs::AbstractVector, ys::AbstractVector) where T
    n = length(xs)
    N = length(out)
    if n == 1
        fill!(out, T(ys[1])); return out
    end
    h = T[xs[i + 1] - xs[i] for i in 1:n-1]
    M = zeros(T, n)                         # second derivatives, natural ends
    if n > 2
        # tridiagonal system for M[2:n-1]
        a = T[h[i] for i in 1:n-2]           # sub-diagonal (unused first)
        b = T[2 * (h[i] + h[i + 1]) for i in 1:n-2]
        c = T[h[i + 1] for i in 1:n-2]
        d = T[6 * ((ys[i + 2] - ys[i + 1]) / h[i + 1] - (ys[i + 1] - ys[i]) / h[i]) for i in 1:n-2]
        for i in 2:n-2                         # Thomas forward sweep
            f = a[i] / b[i - 1]
            b[i] -= f * c[i - 1]
            d[i] -= f * d[i - 1]
        end
        M[n - 1] = d[n - 2] / b[n - 2]
        for i in n-3:-1:1
            M[i + 1] = (d[i] - c[i] * M[i + 2]) / b[i]
        end
    end
    k = 1
    @inbounds for t in 1:N
        while k < n - 1 && t > xs[k + 1]
            k += 1
        end
        x0, x1 = xs[k], xs[k + 1]
        hk = x1 - x0
        A = (x1 - t) / hk; B = (t - x0) / hk
        out[t] = A * ys[k] + B * ys[k + 1] + ((A^3 - A) * M[k] + (B^3 - B) * M[k + 1]) * hk^2 / 6
    end
    return out
end

# strict local maxima and minima (plateaus take their first sample)
function _extrema(h::AbstractVector)
    mx = Int[]; mn = Int[]
    @inbounds for i in 2:length(h)-1
        if h[i] > h[i - 1] && h[i] ≥ h[i + 1]
            push!(mx, i)
        elseif h[i] < h[i - 1] && h[i] ≤ h[i + 1]
            push!(mn, i)
        end
    end
    return mx, mn
end

# envelope through extrema `p`, mirrored about both ends (two extrema each side)
function _envelope!(out::AbstractVector{T}, h::AbstractVector, p::Vector{Int}) where T
    N = length(h)
    k = min(2, length(p))
    left  = [2 - p[i] for i in k:-1:1]              # mirrored about sample 1
    right = [2N - p[end - i + 1] for i in 1:k]      # mirrored about sample N
    xs = vcat(left, p, right)
    ys = vcat(h[p[k:-1:1]], h[p], h[p[end:-1:end-k+1]])
    # keep the knots strictly increasing (mirrors of extrema at the very ends)
    keep = [1; findall(>(0), diff(xs)) .+ 1]
    return _spline!(out, xs[keep], ys[keep])
end

"""
    emd(x; max_imfs=10, max_sift=50, sd=0.2) -> (imfs, residual)

Empirical mode decomposition (Huang et al. 1998): intrinsic mode functions
are sifted out of the signal one at a time by subtracting the mean of the
cubic-spline envelopes through the maxima and the minima (mirrored at both
ends), until the normalised squared difference between two sifts is below
`sd` or `max_sift` sifts were done. Decomposition stops when the residual
has fewer than three extrema or `max_imfs` were extracted. `imfs` is
`nimfs × length(x)`, the fastest oscillation first, and
`sum(imfs, dims=1)' + residual == x`.
"""
function emd(x::AbstractVector{<:Real}; max_imfs::Int=10, max_sift::Int=50, sd::Real=0.2)
    T = float(eltype(x))
    N = length(x)
    max_imfs ≥ 1 || throw(ArgumentError("max_imfs must be ≥ 1"))
    r = Vector{T}(x)
    imfs = Vector{Vector{T}}()
    up = Vector{T}(undef, N); lo = Vector{T}(undef, N)
    for _ in 1:max_imfs
        mx, mn = _extrema(r)
        length(mx) + length(mn) < 3 && break
        h = copy(r)
        for _ in 1:max_sift
            mx, mn = _extrema(h)
            (isempty(mx) || isempty(mn) || length(mx) + length(mn) < 3) && break
            _envelope!(up, h, mx)
            _envelope!(lo, h, mn)
            num = zero(T); den = zero(T)
            @inbounds for i in 1:N
                m = (up[i] + lo[i]) / 2
                num += m^2; den += h[i]^2
                h[i] -= m
            end
            den > 0 && num / den < sd && break
        end
        push!(imfs, h)
        r .-= h
    end
    M = isempty(imfs) ? zeros(T, 0, N) : permutedims(reduce(hcat, imfs))
    return M, r
end

# ---------------------------------------------------------------------------- #
#                           Hilbert-Huang spectrum                             #
# ---------------------------------------------------------------------------- #
struct HhtSetup <: AbstractSetup
    sr       :: Int64
    winsize  :: Int64
    winstep  :: Int64
    offset   :: Int64
    spectrum :: Base.Callable
    nimfs    :: Int64
end

"""
    Hht{T} <: AbstractSpectrogram

Hilbert-Huang spectrum: the instantaneous power of every intrinsic mode
function ([`emd`](@ref)) accumulated at its instantaneous frequency on a
linear grid, pooled on the frames of a [`Frames`](@ref) object. See
[`Hht(frames; kwargs...)`](@ref Hht(::Frames)).
"""
struct Hht{T<:AudioData} <: AbstractSpectrogram
    spec   :: Matrix{T}
    freq   :: Vector{T}
    frames :: Frames{T}
    info   :: HhtSetup
    imfs   :: Matrix{T}
end
@pooled_frontend Hht

"""
    get_imfs(h::Hht) -> Matrix

The intrinsic mode functions the spectrum was built from, `nimfs × samples`.
"""
get_imfs(h::Hht) = h.imfs
Base.show(io::IO, h::Hht{T}) where T =
    print(io, "Hht{$T}($(size(h.spec, 2)) frames × $(size(h.spec, 1)) bins, $(h.info.nimfs) IMFs)")

"""
    Hht(frames::Frames; nbins=winsize ÷ 2 + 1, max_imfs=10, max_sift=50, sd=0.2, spectrum=power) -> Hht

Hilbert-Huang spectrum of the signal of `frames` (Huang et al. 1998;
audioFlux `HHT`): the signal is decomposed by [`emd`](@ref), every IMF's
analytic signal gives an instantaneous amplitude `a` and frequency `f`
(the phase difference of consecutive samples), and `a²` (or `a` for
`spectrum=magnitude`) is added to the bin of `f` on `nbins` bins from 0 to
`sr/2`, averaged over each frame with the frame window.
"""
function Hht(frames::Frames{T}; nbins::Int=get_winsize(frames) ÷ 2 + 1, max_imfs::Int=10,
             max_sift::Int=50, sd::Real=0.2, spectrum::Base.Callable=power) where T
    _check_spectrum(spectrum)
    nbins ≥ 2 || throw(ArgumentError("nbins must be ≥ 2"))
    sr = get_sr(frames)
    x  = get_signal(frames)
    N  = length(x)
    imfs, _ = emd(x; max_imfs, max_sift, sd)
    w = get_window(frames); ws = length(w); wsum = sum(w)
    spec = zeros(T, nbins, length(frames))
    bin = Vector{Int}(undef, N); val = Vector{T}(undef, N)
    for k in axes(imfs, 1)
        z = hilbert(view(imfs, k, :))
        @inbounds for n in 1:N
            d = n == 1 ? (N > 1 ? angle(z[2]) - angle(z[1]) : zero(T)) : angle(z[n]) - angle(z[n - 1])
            f = abs(rem2pi(d, RoundNearest)) * sr / (2π)
            bin[n] = clamp(round(Int, f / (sr / 2) * (nbins - 1)) + 1, 1, nbins)
            val[n] = T(spectrum(z[n]))
        end
        @inbounds for (j, st) in enumerate(frames.starts)
            for i in 1:ws
                n = st + i - 1
                spec[bin[n], j] += w[i] * val[n] / wsum
            end
        end
    end
    freq = T[(k - 1) * sr / (2 * (nbins - 1)) for k in 1:nbins]
    info = HhtSetup(sr, get_winsize(frames), get_step(frames), get_offset(frames), spectrum, size(imfs, 1))
    return Hht{T}(spec, freq, frames, info, Matrix{T}(imfs))
end

# ---------------------------------------------------------------------------- #
#                         empirical wavelet transform                          #
# ---------------------------------------------------------------------------- #
_ewt_beta(x) = x ≤ 0 ? 0.0 : x ≥ 1 ? 1.0 : x^4 * (35 - 84x + 70x^2 - 20x^3)

# Meyer-type filters of Gilles (2013) on the angular grid ω ∈ [0, π] of an
# N-point FFT (symmetric in ω), for the boundaries 0 < ω₁ < … < ω_{n-1} < π
function _ewt_filters(N::Int, bounds::Vector{Float64}, γ::Float64)
    ω = [2π * min(k, N - k) / N for k in 0:N-1]
    nb = length(bounds) + 1
    F = zeros(nb, N)
    for (i, w) in enumerate(ω)
        # scaling function below ω₁
        w1 = bounds[1]
        F[1, i] = w ≤ (1 - γ) * w1 ? 1.0 :
                  w ≤ (1 + γ) * w1 ? cos(π / 2 * _ewt_beta((w - (1 - γ) * w1) / (2γ * w1))) : 0.0
        for n in 2:nb
            a = bounds[n - 1]
            b = n ≤ length(bounds) ? bounds[n] : Inf
            v = 0.0
            if (1 + γ) * a ≤ w ≤ (1 - γ) * b
                v = 1.0
            elseif (1 - γ) * b ≤ w ≤ (1 + γ) * b
                v = cos(π / 2 * _ewt_beta((w - (1 - γ) * b) / (2γ * b)))
            elseif (1 - γ) * a ≤ w ≤ (1 + γ) * a
                v = sin(π / 2 * _ewt_beta((w - (1 - γ) * a) / (2γ * a)))
            end
            F[n, i] = v
        end
    end
    return F
end

"""
    ewt(x, sr; nbands=5) -> (components, bounds)

Empirical wavelet transform (Gilles 2013; audioFlux `EWT`): the Fourier
spectrum of the signal is segmented at the midpoints between its `nbands`
largest local maxima, a Littlewood-Paley (Meyer-type) filter bank is built
on those segments (transition ratio `γ` just below the tightest segment
allows) and the signal is filtered into `nbands` components
(`nbands × length(x)`, lowest band first). `bounds` are the segment
boundaries in Hz. The filters form a tight frame: `Σ |filters|² = 1`.
"""
function ewt(x::AbstractVector{<:Real}, sr::Int; nbands::Int=5)
    T = float(eltype(x))
    N = length(x)
    nbands ≥ 2 || throw(ArgumentError("nbands must be ≥ 2"))
    X = fft(Vector{Complex{T}}(x))
    half = N ÷ 2 + 1
    mag = abs.(X[1:half])
    peaks = [i for i in 2:half-1 if mag[i] > mag[i - 1] && mag[i] ≥ mag[i + 1]]
    length(peaks) ≥ nbands || throw(ArgumentError(
        "the spectrum has only $(length(peaks)) local maxima; reduce nbands"))
    top = sort(peaks[sortperm(mag[peaks], rev=true)[1:nbands]])
    bounds = [π * ((top[i] - 1) + (top[i + 1] - 1)) / N for i in 1:nbands-1]   # rad/sample
    edges = vcat(0.0, bounds, π)
    γ = 0.99 * minimum((edges[i + 1] - edges[i]) / (edges[i + 1] + edges[i]) for i in 2:length(edges)-1)
    F = _ewt_filters(N, bounds, γ)
    comps = Matrix{T}(undef, nbands, N)
    for n in 1:nbands
        comps[n, :] .= real.(ifft(X .* F[n, :]))
    end
    return comps, T.(bounds .* sr ./ (2π))
end

"""
    Ewt(frames::Frames; nbands=5, spectrum=power) -> DiscreteWavelet

Empirical wavelet transform ([`ewt`](@ref)) of the signal of `frames`, its
components pooled on the frames (power or magnitude averaged over each
frame with the frame window); `get_freq` gives the centre of every
adaptive band.
"""
function Ewt(frames::Frames{T}; nbands::Int=5, spectrum::Base.Callable=power) where T
    _check_spectrum(spectrum)
    sr = get_sr(frames)
    comps, bounds = ewt(get_signal(frames), sr; nbands)
    edges = vcat(zero(T), bounds, T(sr / 2))
    freq = T[(edges[i] + edges[i + 1]) / 2 for i in 1:nbands]
    info = DiscreteWaveletSetup(sr, get_winsize(frames), get_step(frames), get_offset(frames),
                                spectrum, :ewt, "empirical", nbands)
    return DiscreteWavelet{T}(_pool_rows(comps, frames, spectrum), freq, frames, info)
end

for S in (:Hht, :Ewt)
    @eval begin
        $S(audio::AbstractVecOrMat{<:Real}, sr::Int; winsize::Int=sr ≤ 8000 ? 256 : 512, winstep::Int=winsize ÷ 2,
           type::Base.Callable=rect, periodic::Bool=true, center::Bool=false, pad_mode::Symbol=:constant, kwargs...) =
            $S(Frames(audio, sr; winsize, winstep, type, periodic, center, pad_mode); kwargs...)
        $S(a::AudioFile; kwargs...) = $S(get_data(a), get_sr(a); kwargs...)
    end
end
