# ---------------------------------------------------------------------------- #
#                                rectifications                                #
# ---------------------------------------------------------------------------- #
"""
    mlog(x)

Base-10 logarithm rectification, MATLAB's `Rectification="log"`.
"""
mlog(x) = log10(x)
mlog(x, y) = x * log10.(y)

"""
    nlog(x)

Natural logarithm rectification (HTK, Kaldi, ETSI, python_speech_features).
"""
nlog(x) = log(x)
nlog(x, y) = x * log.(y)

"""
    cubic_root(x)

Cubic-root rectification, MATLAB's `Rectification="cubic-root"`.
"""
cubic_root(x) = cbrt(x)
cubic_root(x, y) = x * cbrt.(y)

"""
    db(x)

Decibel rectification `10 log10(x)` (librosa's `power_to_db`). Combine with
the `top_db` keyword of [`Mfcc`](@ref) to clip the dynamic range like librosa.
"""
db(x) = 10 * log10(x)
db(x, y) = x * (10 .* log10.(y))

# ---------------------------------------------------------------------------- #
#                                 DCT matrices                                 #
# ---------------------------------------------------------------------------- #
"""
    dct_ortho(T, N) -> Matrix{T}

Orthonormal DCT-II matrix: row 0 scaled by `sqrt(1/N)`, rows `k ≥ 1` by
`sqrt(2/N)`. Used by MATLAB, Kaldi, librosa (`norm="ortho"`) and
python_speech_features.
"""
function dct_ortho(::Type{T}, N::Int) where {T<:AbstractFloat}
    D = Matrix{T}(undef, N, N)
    s0, s1 = sqrt(1 / N), sqrt(2 / N)
    @inbounds for n in 1:N
        D[1, n] = T(s0)
        for k in 2:N
            D[k, n] = T(s1 * cos(π * (k - 1) * (n - 0.5) / N))
        end
    end
    return D
end

"""
    dct_htk(T, N) -> Matrix{T}

HTK's DCT-II matrix: every row, including row 0, scaled by `sqrt(2/N)`.
"""
function dct_htk(::Type{T}, N::Int) where {T<:AbstractFloat}
    D = dct_ortho(T, N)
    D[1, :] .= T(sqrt(2 / N))
    return D
end

"""
    dct_plain(T, N) -> Matrix{T}

Unscaled DCT-II matrix `cos(π k (n + 0.5) / N)` (ETSI ES 201 108).
"""
function dct_plain(::Type{T}, N::Int) where {T<:AbstractFloat}
    D = Matrix{T}(undef, N, N)
    @inbounds for n in 1:N, k in 1:N
        D[k, n] = T(cos(π * (k - 1) * (n - 0.5) / N))
    end
    return D
end

# ---------------------------------------------------------------------------- #
#                                energy sources                                #
# ---------------------------------------------------------------------------- #
"""
    raw_energy(s) -> Vector

Per-frame energy of the raw signal, `sum(x.^2)` before pre-emphasis and
windowing (HTK `RAWENERGY=T`, Kaldi `raw_energy=true`). Pass as
`energy=raw_energy` to [`Mfcc`](@ref).
"""
raw_energy(s::AbstractAudioSpectrum) = get_energy(s)

"""
    spectrum_energy(s) -> Vector

Per-frame energy taken as the sum of the front-end spectrum bins
(python_speech_features `appendEnergy`). Pass as `energy=spectrum_energy`
to [`Mfcc`](@ref).
"""
spectrum_energy(s::AbstractAudioSpectrum) = vec(sum(get_spec(get_frontend(s)), dims=1))

# ---------------------------------------------------------------------------- #
#                                    info                                      #
# ---------------------------------------------------------------------------- #
struct MfccSetup <: AbstractSetup
    sr            :: Int64
    ncoeffs       :: Int64
    rect          :: Base.Callable
    dither        :: Bool
    floor         :: Float64
    dct           :: Base.Callable
    lifter        :: Int64
    lifter_offset :: Int64
    first         :: Int64
    energy        :: Maybe{Base.Callable}
    energy_mode   :: Symbol
    energy_floor  :: Float64
    top_db        :: Maybe{Float64}
end

# ---------------------------------------------------------------------------- #
#                                 mfcc struct                                  #
# ---------------------------------------------------------------------------- #
"""
    Mfcc{F,T} <: AbstractCepstrum

Mel-frequency cepstral coefficients, stored as `ncoeffs × frames`. `F` is the
type of the spectrogram they were computed from. See [`Mfcc(spec; kwargs...)`](@ref Mfcc(::AbstractSpectrogram)).
"""
struct Mfcc{F,T<:AbstractFloat} <: AbstractCepstrum
    spec   :: Matrix{T}
    parent :: F
    info   :: MfccSetup
end

"""
    Gtcc{F,T} <: AbstractCepstrum

Gammatone cepstral coefficients (MATLAB's `gtcc`), the cepstrum of an
[`ErbSpec`](@ref). See [`Gtcc`](@ref Gtcc(::ErbSpec)).
"""
struct Gtcc{F,T<:AbstractFloat} <: AbstractCepstrum
    spec   :: Matrix{T}
    parent :: F
    info   :: MfccSetup
end

# ---------------------------------------------------------------------------- #
#                                    methods                                   #
# ---------------------------------------------------------------------------- #
Base.eltype(::Mfcc{F,T}) where {F,T} = T
Base.eltype(::Gtcc{F,T}) where {F,T} = T

"""
    get_data(m::AbstractCepstrum) -> AbstractMatrix

Coefficients transposed to `frames × ncoeffs` (MATLAB orientation).
"""
@inline get_data(m::AbstractCepstrum)   = m.spec'
@inline get_spec(m::AbstractCepstrum)   = m.spec
@inline get_setup(m::AbstractCepstrum)  = m.info
@inline get_sr(m::AbstractCepstrum)     = m.info.sr
@inline get_parent(m::AbstractCepstrum) = m.parent

"""
    get_ncoeffs(m::AbstractCepstrum) -> Int

Number of coefficients per frame (including an appended energy, if any).
"""
@inline get_ncoeffs(m::AbstractCepstrum) = size(m.spec, 1)

function Base.show(io::IO, m::Mfcc{F,T}) where {F,T}
    nc, nf = size(m.spec)
    print(io, "Mfcc{$(nameof(F)),$T}($nf frames × $nc coeffs, rect=$(m.info.rect))")
end
function Base.show(io::IO, m::Gtcc{F,T}) where {F,T}
    nc, nf = size(m.spec)
    print(io, "Gtcc{$(nameof(F)),$T}($nf frames × $nc coeffs, rect=$(m.info.rect))")
end

function _show_cepstrum(io::IO, name::String, m::AbstractCepstrum)
    nc, nf = size(m.spec)
    i = m.info
    println(io, name)
    println(io, "  Dimensions:")
    println(io, "    Frames:        $nf")
    println(io, "    Coefficients:  $nc (from index $(i.first))")
    println(io, "  Configuration:")
    println(io, "    Sample rate:   $(i.sr) Hz")
    println(io, "    Rectification: $(i.rect)")
    println(io, "    DCT:           $(i.dct)")
    iszero(i.lifter) || println(io, "    Lifter:        $(i.lifter) (offset $(i.lifter_offset))")
    isnothing(i.energy) || println(io, "    Energy:        $(i.energy) ($(i.energy_mode))")
    print(io,   "    Dither:        $(i.dither)")
end
Base.show(io::IO, ::MIME"text/plain", m::Mfcc{F,T}) where {F,T} = _show_cepstrum(io, "Mfcc{$(nameof(F)),$T}", m)
Base.show(io::IO, ::MIME"text/plain", m::Gtcc{F,T}) where {F,T} = _show_cepstrum(io, "Gtcc{$(nameof(F)),$T}", m)

# ---------------------------------------------------------------------------- #
#                                   cepstrum                                   #
# ---------------------------------------------------------------------------- #
function _cepstrum(
    spec          :: AbstractSpectrogram;
    ncoeffs       :: Int64=get_nbins(spec) ÷ 2,
    rect          :: Base.Callable=mlog,
    dither        :: Bool=false,
    floor         :: Maybe{Real}=nothing,
    dct           :: Base.Callable=dct_ortho,
    lifter        :: Int64=0,
    lifter_offset :: Int64=0,
    first         :: Int64=0,
    energy        :: Maybe{Base.Callable}=nothing,
    energy_mode   :: Symbol=:replace,
    energy_floor  :: Maybe{Real}=nothing,
    top_db        :: Maybe{Real}=nothing,
)
    T  = eltype(spec)
    S  = get_spec(spec)                 # (nbands × nframes)
    nb, nf = size(S)

    ncoeffs ≥ 1 || throw(ArgumentError("ncoeffs must be ≥ 1, got $ncoeffs"))
    first ≥ 0   || throw(ArgumentError("first must be ≥ 0, got $first"))
    first + ncoeffs ≤ nb || throw(ArgumentError(
        "first + ncoeffs = $(first + ncoeffs) exceeds the number of bands ($nb)"))
    lifter ≥ 0  || throw(ArgumentError("lifter must be ≥ 0, got $lifter"))
    energy_mode in (:replace, :append, :prepend) || throw(ArgumentError(
        "energy_mode must be :replace, :append or :prepend, got $energy_mode"))
    (isnothing(energy) || energy_mode != :replace || first == 0) || throw(ArgumentError(
        "energy_mode=:replace needs first=0 so that C0 is present"))

    fl = isnothing(floor) ? (dither ? T(1e-8) : floatmin(T)) : T(floor)

    # rectification on a floored copy (the input spectrogram is never touched)
    R = Matrix{T}(undef, nb, nf)
    @inbounds for i in eachindex(S)
        R[i] = rect(max(S[i], fl))
    end
    if !isnothing(top_db)
        lo = maximum(R) - T(top_db)
        @inbounds for i in eachindex(R)
            R[i] = max(R[i], lo)
        end
    end

    # DCT and coefficient selection
    D = dct(T, nb)
    C = (@view D[first+1:first+ncoeffs, :]) * R

    # cepstral liftering
    if lifter > 0
        L = T(lifter)
        @inbounds for r in 1:ncoeffs
            i = first + r - 1 + lifter_offset
            g = one(T) + L / 2 * sin(T(π) * i / L)
            @views C[r, :] .*= g
        end
    end

    # log energy
    efl = isnothing(energy_floor) ? eps(T) : T(energy_floor)
    if !isnothing(energy)
        e  = energy(spec)
        le = T[log(max(T(v), efl)) for v in e]
        if energy_mode == :replace
            C[1, :] .= le
        elseif energy_mode == :append
            C = vcat(C, permutedims(le))
        else
            C = vcat(permutedims(le), C)
        end
    end

    info = MfccSetup(get_sr(spec), ncoeffs, rect, dither,
                     Float64(fl), dct, lifter, lifter_offset, first,
                     energy, energy_mode, Float64(efl),
                     isnothing(top_db) ? nothing : Float64(top_db))
    return C, info
end

"""
    Mfcc(spec::AbstractSpectrogram; kwargs...) -> Mfcc

Cepstral coefficients of a band spectrogram, usually a [`MelSpec`](@ref) or
[`BarkSpec`](@ref) (MATLAB's `mfcc` / `cepstralCoefficients`). Every axis on
which the published MFCC variants differ is a keyword, see the
[MFCC variants](@ref mfcc_variants) page for the recipes.

The bands are floored, rectified, transformed with a DCT matrix, cut to
`ncoeffs` coefficients starting at index `first`, liftered and optionally
combined with a log-energy term.

# Keyword Arguments
- `ncoeffs::Int=nbands ÷ 2`: number of coefficients kept
- `rect=mlog`: rectification, `mlog` (log10), `nlog` (ln), `cubic_root` or `db`
- `dither::Bool=false`: floor the bands at `1e-8` instead of `floatmin(T)`
  (MATLAB's `dither`)
- `floor::Real`: explicit band floor (overrides `dither`); Kaldi uses `eps(Float32)`
- `dct=dct_ortho`: DCT matrix, `dct_ortho`, `dct_htk` or `dct_plain`
- `lifter::Int=0`: cepstral lifter length `L`, coefficient `i` is scaled by
  `1 + L/2 sin(π i / L)` (22 in HTK and Kaldi)
- `lifter_offset::Int=0`: added to `i` before liftering (`1` reproduces librosa)
- `first::Int=0`: index of the first coefficient kept (`1` drops C0, HTK default)
- `energy=nothing`: `raw_energy` or `spectrum_energy` to add a natural-log
  energy term
- `energy_mode::Symbol=:replace`: `:replace` C0 with the log energy (Kaldi,
  python_speech_features), `:append` it as a last coefficient (HTK `_E`) or
  `:prepend` it as a first one (audioFlux `CepstralEnergyType.APPEND`)
- `energy_floor::Real=eps(T)`: floor applied to the energy before its log
  (`exp(-50)` in ETSI)
- `top_db::Real`: clip the rectified bands to `max - top_db` (librosa's
  `power_to_db(top_db=80)`), for `rect=db`

# Examples
```julia
mel  = MelSpec(stft; nbands=32)
mfcc = Mfcc(mel; ncoeffs=13, rect=mlog)                   # MATLAB
mfcc = Mfcc(mel; ncoeffs=13, rect=nlog, lifter=22, energy=raw_energy)   # Kaldi-like
mfcc = Mfcc(mel; ncoeffs=12, first=1, rect=nlog, dct=dct_htk, lifter=22) # HTK-like
```
"""
function Mfcc(spec::AbstractSpectrogram; kwargs...)
    coeffs, info = _cepstrum(spec; kwargs...)
    Mfcc{typeof(spec),eltype(spec)}(coeffs, spec, info)
end

"""
    Gtcc(erb::ErbSpec; kwargs...) -> Gtcc

Gammatone cepstral coefficients (MATLAB's `gtcc`): the cepstrum of an
[`ErbSpec`](@ref), with the keywords of [`Mfcc`](@ref).
"""
function Gtcc(erb::ErbSpec; kwargs...)
    coeffs, info = _cepstrum(erb; kwargs...)
    Gtcc{typeof(erb),eltype(erb)}(coeffs, erb, info)
end

function Gtcc(spec::AbstractSpectrogram; kwargs...)
    throw(ArgumentError(
        "Gtcc requires an ErbSpec (gammatone filterbank) input, got $(typeof(spec)). " *
        "Use Mfcc for MelSpec or BarkSpec inputs."
    ))
end
