# ---------------------------------------------------------------------------- #
#                               chroma filterbank                              #
# ---------------------------------------------------------------------------- #
"""
    hz_to_octs(f; tuning=0, bins_per_octave=12)

Octave number of a frequency relative to `A440 / 16` (librosa `hz_to_octs`).
"""
hz_to_octs(f; tuning::Real=0, bins_per_octave::Int=12) =
    log2(f / (440 / 16 * 2.0^(tuning / bins_per_octave)))

struct ChromaFBankSetup <: AbstractSetup
    sr::Int64
    nchroma::Int64
    tuning::Float64
    ctroct::Float64
    octwidth::Maybe{Float64}
    base_c::Bool
end

"""
    ChromaFBank{T} <: AbstractFBank

A `nchroma × nbins` chroma projection matrix built by [`chroma_fbank`](@ref).
"""
struct ChromaFBank{T<:AudioData} <: AbstractFBank
    fbank::Matrix{T}
    freq::Vector{T}
    setup::ChromaFBankSetup
end

get_data(f::ChromaFBank)   = f.fbank
get_freq(f::ChromaFBank)   = f.freq
get_sr(f::ChromaFBank)     = f.setup.sr
get_nbands(f::ChromaFBank) = f.setup.nchroma
Base.eltype(::ChromaFBank{T}) where T = T
Base.show(io::IO, f::ChromaFBank{T}) where T =
    print(io, "ChromaFBank{$T}($(f.setup.nchroma) × $(size(f.fbank, 2)))")

"""
    chroma_fbank(sr; sfreq, nchroma=12, tuning=0, ctroct=5.0, octwidth=2, base_c=true) -> ChromaFBank
    chroma_fbank(spec::AbstractSpectrogram; kwargs...) -> ChromaFBank

Chroma (pitch-class) filterbank of librosa `filters.chroma`: every bin of
the frequency grid `sfreq` is mapped to a fractional chroma index, weighted
with a Gaussian of width equal to the bin spacing (in chroma units) and,
when `octwidth` is given, with a Gaussian octave weighting centred on
`ctroct`. Columns are L2-normalised; `base_c=true` makes row 1 the pitch
class C.
"""
function chroma_fbank(sr::Int; sfreq::AbstractVector{<:AudioData}, nchroma::Int=12, tuning::Real=0,
                      ctroct::Real=5.0, octwidth::Maybe{Real}=2, base_c::Bool=true)
    T  = eltype(sfreq)
    nf = length(sfreq)
    nf ≥ 2 || throw(ArgumentError("the frequency grid needs at least two bins"))
    # fractional chroma bin of every frequency; DC gets an extrapolated value
    frq = Vector{Float64}(undef, nf)
    for k in 2:nf
        frq[k] = sfreq[k] > 0 ? nchroma * hz_to_octs(Float64(sfreq[k]); tuning, bins_per_octave=nchroma) : NaN
    end
    frq[1] = sfreq[1] > 0 ? nchroma * hz_to_octs(Float64(sfreq[1]); tuning, bins_per_octave=nchroma) : frq[2] - 1.5 * nchroma
    for k in 2:nf
        isnan(frq[k]) && (frq[k] = frq[k+1 ≤ nf ? k+1 : k] - 1.5 * nchroma)
    end
    # bin widths in chroma units
    bw = Vector{Float64}(undef, nf)
    for k in 1:nf-1
        bw[k] = max(frq[k+1] - frq[k], 1.0)
    end
    bw[nf] = 1.0
    half = round(nchroma / 2)
    W = Matrix{Float64}(undef, nchroma, nf)
    for k in 1:nf, c in 1:nchroma
        d = frq[k] - (c - 1)
        d = mod(d + half + 10nchroma, nchroma) - half
        W[c, k] = exp(-0.5 * (2d / bw[k])^2)
    end
    # L2-normalise every column
    for k in 1:nf
        nrm = sqrt(sum(abs2, view(W, :, k)))
        nrm > 0 && (W[:, k] ./= nrm)
    end
    if !isnothing(octwidth)
        for k in 1:nf
            W[:, k] .*= exp(-0.5 * ((frq[k] / nchroma - ctroct) / octwidth)^2)
        end
    end
    base_c && (W = circshift(W, (-3 * (nchroma ÷ 12), 0)))
    return ChromaFBank{T}(Matrix{T}(W), Vector{T}(sfreq),
                          ChromaFBankSetup(sr, nchroma, Float64(tuning), Float64(ctroct),
                                           isnothing(octwidth) ? nothing : Float64(octwidth), base_c))
end

chroma_fbank(s::AbstractSpectrogram; kwargs...) = chroma_fbank(get_sr(s); sfreq=get_freq(s), kwargs...)

# ---------------------------------------------------------------------------- #
#                                    chroma                                    #
# ---------------------------------------------------------------------------- #
struct ChromaSetup <: AbstractSetup
    sr::Int64
    norm::Maybe{Float64}
end

"""
    Chroma{F,B,T} <: AbstractSpectrogram

Chromagram, `nchroma × frames`. `get_freq` returns the pitch-class index
`0:nchroma-1` (0 = C when `base_c`).
"""
struct Chroma{F,B,T<:AudioData} <: AbstractSpectrogram
    spec   :: Matrix{T}
    fbank  :: B
    parent :: F
    info   :: ChromaSetup
end

Base.eltype(::Chroma{F,B,T}) where {F,B,T} = T
get_data(c::Chroma)   = c.spec'
get_spec(c::Chroma)   = c.spec
get_freq(c::Chroma)   = Vector{eltype(c)}(0:size(c.spec, 1)-1)
get_setup(c::Chroma)  = c.info
get_sr(c::Chroma)     = c.info.sr
get_nbands(c::Chroma) = size(c.spec, 1)
get_fbank(c::Chroma)  = c.fbank
get_parent(c::Chroma) = c.parent
Base.show(io::IO, c::Chroma{F,B,T}) where {F,B,T} =
    print(io, "Chroma{$(nameof(F)),$T}($(size(c.spec, 2)) frames × $(size(c.spec, 1)) pitch classes)")

# normalise every column of X by its p-norm (Inf = max); zero columns are left alone
function _normalize_columns!(X::AbstractMatrix{T}, p::Real) where T
    @inbounds for j in axes(X, 2)
        col = view(X, :, j)
        nrm = isinf(p) ? maximum(abs, col) : sum(abs.(col) .^ p)^(1 / p)
        nrm > 0 && (col ./= T(nrm))
    end
    return X
end

"""
    Chroma(spec::AbstractSpectrogram; norm=Inf, kwargs...) -> Chroma
    Chroma(spec::AbstractSpectrogram, fbank::ChromaFBank; norm=Inf) -> Chroma

Chromagram of any front end (librosa `chroma_stft`): the spectrogram is
projected on a [`chroma_fbank`](@ref) and every frame is normalised by its
`norm`-norm (`Inf` = maximum, `1`, `2`, or `nothing` for none). Keywords are
those of `chroma_fbank`. librosa applies this to the power spectrum; pass a
`power` front end to match it.
"""
function Chroma(s::AbstractSpectrogram, fbank::ChromaFBank; norm::Maybe{Real}=Inf)
    T = eltype(s)
    size(get_data(fbank), 2) == get_nbins(s) || throw(DimensionMismatch(
        "chroma filterbank has $(size(get_data(fbank), 2)) bins, the spectrogram $(get_nbins(s))"))
    fb = eltype(fbank) === T ? get_data(fbank) : Matrix{T}(get_data(fbank))
    C  = fb * get_spec(s)
    isnothing(norm) || _normalize_columns!(C, norm)
    return Chroma{typeof(s),typeof(fbank),T}(C, fbank, s, ChromaSetup(get_sr(s), isnothing(norm) ? nothing : Float64(norm)))
end
Chroma(s::AbstractSpectrogram; norm::Maybe{Real}=Inf, kwargs...) =
    Chroma(s, chroma_fbank(s; kwargs...); norm)

# ---------------------------------------------------------------------------- #
#                                   tonnetz                                    #
# ---------------------------------------------------------------------------- #
struct TonnetzSetup <: AbstractSetup
    sr::Int64
end

"""
    Tonnetz{F,T} <: AbstractSpectrogram

Tonal centroid features (Harte, Sandler & Gasser 2006; librosa `tonnetz`),
`6 × frames`: fifths (x, y), minor thirds (x, y), major thirds (x, y).
"""
struct Tonnetz{F,T<:AudioData} <: AbstractSpectrogram
    spec   :: Matrix{T}
    parent :: F
    info   :: TonnetzSetup
end

Base.eltype(::Tonnetz{F,T}) where {F,T} = T
get_data(t::Tonnetz)   = t.spec'
get_spec(t::Tonnetz)   = t.spec
get_freq(t::Tonnetz)   = Vector{eltype(t)}(0:5)
get_setup(t::Tonnetz)  = t.info
get_sr(t::Tonnetz)     = t.info.sr
get_parent(t::Tonnetz) = t.parent
Base.show(io::IO, t::Tonnetz{F,T}) where {F,T} =
    print(io, "Tonnetz{$(nameof(F)),$T}($(size(t.spec, 2)) frames × 6)")

"""
    Tonnetz(chroma::Chroma) -> Tonnetz
    Tonnetz(spec::AbstractSpectrogram; kwargs...) -> Tonnetz

Project an L1-normalised chromagram on the six tonal-centroid basis
functions (radii 1, 1, 0.5 for fifths, minor and major thirds). A
non-chroma spectrogram is first turned into a [`Chroma`](@ref) with `kwargs`.
"""
function Tonnetz(c::Chroma)
    T  = eltype(c)
    nc = get_nbands(c)
    dim = (0:nc-1) .* (12 / nc)
    scale = [7 / 6, 7 / 6, 3 / 2, 3 / 2, 2 / 3, 2 / 3]
    r     = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5]
    Φ = Matrix{T}(undef, 6, nc)
    for i in 1:6, k in 1:nc
        v = scale[i] * dim[k]
        isodd(i) && (v -= 0.5)
        Φ[i, k] = T(r[i] * cospi(v))
    end
    C = copy(get_spec(c))
    _normalize_columns!(C, 1)
    return Tonnetz{typeof(c),T}(Φ * C, c, TonnetzSetup(get_sr(c)))
end
Tonnetz(s::AbstractSpectrogram; kwargs...) = Tonnetz(Chroma(s; kwargs...))

# ---------------------------------------------------------------------------- #
#                               spectral contrast                              #
# ---------------------------------------------------------------------------- #
struct SpectralContrastSetup <: AbstractSetup
    sr::Int64
    nbands::Int64
    fmin::Float64
    quantile::Float64
    linear::Bool
end

"""
    SpectralContrast{F,T} <: AbstractSpectrogram

Octave-based spectral contrast (Jiang et al. 2002; librosa
`spectral_contrast`), `(nbands + 1) × frames`.
"""
struct SpectralContrast{F,T<:AudioData} <: AbstractSpectrogram
    spec   :: Matrix{T}
    parent :: F
    info   :: SpectralContrastSetup
end

Base.eltype(::SpectralContrast{F,T}) where {F,T} = T
get_data(x::SpectralContrast)   = x.spec'
get_spec(x::SpectralContrast)   = x.spec
get_freq(x::SpectralContrast)   = Vector{eltype(x)}(0:size(x.spec, 1)-1)
get_setup(x::SpectralContrast)  = x.info
get_sr(x::SpectralContrast)     = x.info.sr
get_parent(x::SpectralContrast) = x.parent
Base.show(io::IO, x::SpectralContrast{F,T}) where {F,T} =
    print(io, "SpectralContrast{$(nameof(F)),$T}($(size(x.spec, 2)) frames × $(size(x.spec, 1)) bands)")

# magnitude view of any front end
_magnitude_spec(s::AbstractSpectrogram) = get_spectrum(s) === power ? sqrt.(get_spec(s)) : get_spec(s)

"""
    SpectralContrast(spec; nbands=6, fmin=200, quantile=0.02, linear=false) -> SpectralContrast

For every octave band above `fmin` (plus the band below it) the difference
between the mean of the top `quantile` of the sorted magnitudes (peak) and
the mean of the bottom `quantile` (valley), in natural-log units
(`linear=true` returns the raw difference). librosa `spectral_contrast`.
"""
function SpectralContrast(s::AbstractSpectrogram; nbands::Int=6, fmin::Real=200,
                          quantile::Real=0.02, linear::Bool=false)
    T = eltype(s)
    S = _magnitude_spec(s)
    f = get_freq(s)
    sr = get_sr(s)
    fmin > 0 || throw(ArgumentError("fmin must be positive"))
    0 < quantile < 1 || throw(ArgumentError("quantile must be in (0, 1)"))
    nbands ≥ 1 || throw(ArgumentError("nbands must be ≥ 1"))
    octa = vcat(0.0, [Float64(fmin) * 2.0^k for k in 0:nbands])
    octa[end-1] ≤ sr / 2 || throw(ArgumentError("frequency band exceeds Nyquist: nbands=$nbands octaves above fmin=$fmin"))
    nb, nf = size(S)
    out = zeros(T, nbands + 1, nf)
    for k in 0:nbands
        lo, hi = octa[k + 1], octa[k + 2]
        inband = [lo ≤ f[i] ≤ hi for i in 1:nb]
        idx = findall(inband)
        isempty(idx) && continue
        k > 0 && idx[1] > 1 && (inband[idx[1] - 1] = true)          # one bin below the band
        k == nbands && (inband[idx[end]+1:end] .= true)              # everything above the top band
        sel = findall(inband)
        q = max(round(Int, quantile * length(sel)), 1)
        k < nbands && length(sel) > 1 && (sel = sel[1:end-1])        # drop the last bin
        q = min(q, length(sel))
        for j in 1:nf
            v = sort(S[sel, j])
            valley = sum(view(v, 1:q)) / q
            peak   = sum(view(v, length(v)-q+1:length(v))) / q
            out[k + 1, j] = linear ? peak - valley : log(peak + T(1e-6)) - log(valley + T(1e-6))
        end
    end
    return SpectralContrast{typeof(s),T}(out, s, SpectralContrastSetup(sr, nbands, Float64(fmin), Float64(quantile), linear))
end

# ---------------------------------------------------------------------------- #
#                                 poly features                                #
# ---------------------------------------------------------------------------- #
struct PolySetup <: AbstractSetup
    sr::Int64
    order::Int64
end

"""
    PolyFeatures{F,T} <: AbstractSpectrogram

Coefficients of a polynomial fitted to every frame of a spectrogram against
frequency (librosa `poly_features`), `(order + 1) × frames`, highest degree
first.
"""
struct PolyFeatures{F,T<:AudioData} <: AbstractSpectrogram
    spec   :: Matrix{T}
    parent :: F
    info   :: PolySetup
end

Base.eltype(::PolyFeatures{F,T}) where {F,T} = T
get_data(x::PolyFeatures)   = x.spec'
get_spec(x::PolyFeatures)   = x.spec
get_freq(x::PolyFeatures)   = Vector{eltype(x)}(0:size(x.spec, 1)-1)
get_setup(x::PolyFeatures)  = x.info
get_sr(x::PolyFeatures)     = x.info.sr
get_parent(x::PolyFeatures) = x.parent
Base.show(io::IO, x::PolyFeatures{F,T}) where {F,T} =
    print(io, "PolyFeatures{$(nameof(F)),$T}($(size(x.spec, 2)) frames, order $(x.info.order))")

"""
    PolyFeatures(spec; order=1) -> PolyFeatures

Least-squares polynomial fit of degree `order` of each frame's bins against
their frequency (librosa `poly_features`). Row 1 is the highest-degree
coefficient.
"""
function PolyFeatures(s::AbstractSpectrogram; order::Int=1)
    T = eltype(s)
    S = get_spec(s)
    f = Vector{T}(get_freq(s))
    order ≥ 0 || throw(ArgumentError("order must be ≥ 0"))
    V = Matrix{T}(undef, length(f), order + 1)
    for (i, x) in enumerate(f), d in 0:order
        V[i, d + 1] = x^(order - d)
    end
    coeffs = V \ S
    return PolyFeatures{typeof(s),T}(Matrix{T}(coeffs), s, PolySetup(get_sr(s), order))
end
