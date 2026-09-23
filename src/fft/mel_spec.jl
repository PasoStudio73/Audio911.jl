# ---------------------------------------------------------------------------- #
#                                    info                                      #
# ---------------------------------------------------------------------------- #
struct MelSpecSetup <: AbstractSetup
    sr       :: Int64
    win_norm :: Bool
end

# apply a filterbank to a front end; returns the (nbands × nframes) product
function _apply_fbank(s::AbstractSpectrogram, fbank::AbstractFBank, win_norm::Bool)
    T = eltype(s)
    _check_bins(s, fbank)
    fb = get_data(fbank)
    factor = win_norm ? get_winnorm(s) : one(T)
    if eltype(fb) !== T || factor != one(T)
        fb = Matrix{T}(fb)
        factor == one(T) || (fb .*= factor)
    end
    return fb * get_spec(s)
end

# ---------------------------------------------------------------------------- #
#                            mel spectrogram struct                            #
# ---------------------------------------------------------------------------- #
"""
    MelSpec{F,B,T} <: AbstractSpectrogram

Mel spectrogram: a front end multiplied by a triangular mel filterbank
(MATLAB's `melSpectrum`). `F` is the front-end type, `B` the filterbank type.

Build one with [`MelSpec(spec; kwargs...)`](@ref MelSpec(::AbstractSpectrogram))
or [`MelSpec(spec, fbank; win_norm)`](@ref MelSpec(::AbstractSpectrogram, ::AbstractFBank)).
"""
struct MelSpec{F,B,T} <: AbstractSpectrogram
    spec   :: Matrix{T}
    fbank  :: B
    parent :: F
    info   :: MelSpecSetup
end

# ---------------------------------------------------------------------------- #
#                                    methods                                   #
# ---------------------------------------------------------------------------- #
Base.eltype(::MelSpec{F,B,T}) where {F,B,T} = T

"""
    get_data(m::MelSpec) -> AbstractMatrix

The mel spectrogram transposed to `frames × bands` (MATLAB orientation).
"""
@inline get_data(m::MelSpec)  = m.spec'
@inline get_spec(m::MelSpec)  = m.spec

"""
    get_freq(m::MelSpec) -> Vector

Centre frequency of every band in Hz.
"""
@inline get_freq(m::MelSpec)  = get_freq(m.fbank)
@inline get_freqrange(m::MelSpec) = get_freqrange(m.fbank)
@inline get_setup(m::MelSpec) = m.info
@inline get_sr(m::MelSpec)    = m.info.sr
@inline get_nbands(m::MelSpec) = get_nbands(m.fbank)
@inline get_parent(m::MelSpec) = m.parent

"""
    get_fbank(m) -> FBank

The filterbank a filterbank spectrogram was built with.
"""
@inline get_fbank(m::MelSpec) = m.fbank

# ---------------------------------------------------------------------------- #
#                                     show                                     #
# ---------------------------------------------------------------------------- #
function Base.show(io::IO, m::MelSpec{F,B,T}) where {F,B,T}
    nbands, nframes = size(m.spec)
    print(io, "MelSpec{$(nameof(F)),$T}($nframes frames × $nbands bands, sr=$(m.info.sr) Hz, win_norm=$(m.info.win_norm))")
end

function Base.show(io::IO, ::MIME"text/plain", m::MelSpec{F,B,T}) where {F,B,T}
    nbands, nframes = size(m.spec)
    freq_range = extrema(get_freq(m))
    println(io, "MelSpec{$(nameof(F)),$T}")
    println(io, "  Dimensions:")
    println(io, "    Frames:     $nframes")
    println(io, "    Mel bands:  $nbands")
    println(io, "  Configuration:")
    println(io, "    Sample rate:        $(m.info.sr) Hz")
    println(io, "    Scale:              $(get_scale(m.fbank))")
    println(io, "    Frequency range:    $(round(freq_range[1], digits=1)) - $(round(freq_range[2], digits=1)) Hz")
    print(io,   "    Window normalized:  $(m.info.win_norm)")
end

# ---------------------------------------------------------------------------- #
#                                 get mel spec                                 #
# ---------------------------------------------------------------------------- #
"""
    MelSpec(spec::AbstractSpectrogram, fbank::AbstractFBank; win_norm=false) -> MelSpec

Apply a pre-designed filterbank to a front end. The filterbank must have been
designed on the frequency grid of `spec` (`auditory_fbank(spec; ...)`).

- `win_norm::Bool=false`: multiply the filterbank by the window normalisation
  of the front end (`1/sum(w)^2` for a power spectrum, `1/sum(w)` for a
  magnitude spectrum; MATLAB `WindowNormalization`)
"""
function MelSpec(
    s        :: AbstractSpectrogram,
    fbank    :: AbstractFBank;
    win_norm :: Bool=false
)
    spec = _apply_fbank(s, fbank, win_norm)
    info = MelSpecSetup(get_sr(s), win_norm)
    return MelSpec{typeof(s),typeof(fbank),eltype(s)}(spec, fbank, s, info)
end

"""
    MelSpec(spec::AbstractSpectrogram; win_norm=true, kwargs...) -> MelSpec

Design a mel filterbank on the grid of `spec` with [`auditory_fbank`](@ref)
and apply it. `spec` can be any front end (`Stft`, `Cwt`, ...).

# Keyword Arguments
- `win_norm::Bool=true`: window normalisation
- `nbands::Int=26`: number of bands
- `scale=htk`: `htk` or `slaney` mel scale (use [`BarkSpec`](@ref) for bark)
- `norm=bandwidth`: `bandwidth`, `area` or `none_norm`
- `domain=:linear`: `:linear` or `:warped` triangle design
- `freqrange::FreqRange=(0, sr÷2)`

# Examples
```julia
stft = Stft(audio; winsize=512, winstep=256, type=hamming)
mel  = MelSpec(stft; nbands=32, scale=slaney, freqrange=(100, 8000))

cwt  = Cwt(audio; winsize=512, winstep=256)
mel  = MelSpec(cwt; nbands=32)          # same call on a wavelet front end
```
"""
function MelSpec(
    s        :: AbstractSpectrogram;
    win_norm :: Bool=true,
    kwargs...
)
    scale = get(kwargs, :scale, htk)
    scale in (htk, slaney) || throw(ArgumentError(
        "MelSpec only supports `htk` or `slaney` scale, got `$(nameof(scale))`."
    ))
    fbank = auditory_fbank(get_sr(s); sfreq=get_freq(s), kwargs...)
    MelSpec(s, fbank; win_norm)
end

# ---------------------------------------------------------------------------- #
#                                  bark spec                                   #
# ---------------------------------------------------------------------------- #
"""
    BarkSpec{F,B,T} <: AbstractSpectrogram

Bark spectrogram (MATLAB's `barkSpectrum`): a [`MelSpec`](@ref) whose
filterbank is designed on the bark scale.
"""
struct BarkSpec{F,B,T} <: AbstractSpectrogram
    mel :: MelSpec{F,B,T}
end

"""
    BarkSpec(spec::AbstractSpectrogram, fbank::AbstractFBank; win_norm=false) -> BarkSpec
    BarkSpec(spec::AbstractSpectrogram; win_norm=true, kwargs...) -> BarkSpec

Bark-scale filterbank spectrogram. Keywords are those of [`MelSpec`](@ref)
except `scale`, which is always `bark`.
"""
BarkSpec(s::AbstractSpectrogram, fbank::AbstractFBank; win_norm::Bool=false) =
    BarkSpec(MelSpec(s, fbank; win_norm))

function BarkSpec(s::AbstractSpectrogram; win_norm::Bool=true, kwargs...)
    haskey(kwargs, :scale) && throw(ArgumentError(
        "BarkSpec does not accept a `scale` keyword; the Bark scale is always used. " *
        "Use MelSpec for htk/slaney scales."
    ))
    fbank = auditory_fbank(get_sr(s); sfreq=get_freq(s), scale=bark, kwargs...)
    BarkSpec(s, fbank; win_norm)
end

Base.eltype(::BarkSpec{F,B,T}) where {F,B,T} = T
@inline get_data(b::BarkSpec)      = get_data(b.mel)
@inline get_spec(b::BarkSpec)      = get_spec(b.mel)
@inline get_freq(b::BarkSpec)      = get_freq(b.mel)
@inline get_freqrange(b::BarkSpec) = get_freqrange(b.mel)
@inline get_setup(b::BarkSpec)     = get_setup(b.mel)
@inline get_sr(b::BarkSpec)        = get_sr(b.mel)
@inline get_nbands(b::BarkSpec)    = get_nbands(b.mel)
@inline get_fbank(b::BarkSpec)     = get_fbank(b.mel)
@inline get_parent(b::BarkSpec)    = get_parent(b.mel)

function Base.show(io::IO, b::BarkSpec{F,B,T}) where {F,B,T}
    nframes, nbands = size(get_data(b))
    print(io, "BarkSpec{$(nameof(F)),$T}($nframes frames × $nbands bands, sr=$(get_sr(b)) Hz)")
end

function Base.show(io::IO, ::MIME"text/plain", b::BarkSpec{F,B,T}) where {F,B,T}
    nframes, nbands = size(get_data(b))
    freq_range = extrema(get_freq(b))
    println(io, "BarkSpec{$(nameof(F)),$T}")
    println(io, "  Dimensions:")
    println(io, "    Frames:     $nframes")
    println(io, "    Bark bands: $nbands")
    println(io, "  Configuration:")
    println(io, "    Sample rate:       $(get_sr(b)) Hz")
    println(io, "    Frequency range:   $(round(freq_range[1], digits=1)) - $(round(freq_range[2], digits=1)) Hz")
    print(io,   "    Window normalized: $(get_setup(b).win_norm)")
end

# ---------------------------------------------------------------------------- #
#                                   erb spec                                   #
# ---------------------------------------------------------------------------- #
struct ErbSpecSetup <: AbstractSetup
    sr       :: Int64
    win_norm :: Bool
end

"""
    ErbSpec{F,B,T} <: AbstractSpectrogram

ERB spectrogram (MATLAB's `erbSpectrum`): a front end multiplied by a
gammatone filterbank designed with [`gammatone_fbank`](@ref).
"""
struct ErbSpec{F,B,T} <: AbstractSpectrogram
    spec   :: Matrix{T}
    fbank  :: B
    parent :: F
    info   :: ErbSpecSetup
end

"""
    ErbSpec(spec::AbstractSpectrogram, fbank::AbstractFBank; win_norm=false) -> ErbSpec
    ErbSpec(spec::AbstractSpectrogram; win_norm=true, kwargs...) -> ErbSpec

Gammatone (ERB-scale) filterbank spectrogram of any front end. The second
form designs the filterbank on the grid of `spec`; keywords `nbands`, `norm`
and `freqrange` are those of [`gammatone_fbank`](@ref).

```julia
erb  = ErbSpec(stft; nbands=34, freqrange=(50, 8000))
gtcc = Gtcc(erb; ncoeffs=13)
```
"""
function ErbSpec(
    s        :: AbstractSpectrogram,
    fbank    :: AbstractFBank;
    win_norm :: Bool=false
)
    spec = _apply_fbank(s, fbank, win_norm)
    info = ErbSpecSetup(get_sr(s), win_norm)
    return ErbSpec{typeof(s),typeof(fbank),eltype(s)}(spec, fbank, s, info)
end

function ErbSpec(s::AbstractSpectrogram; win_norm::Bool=true, kwargs...)
    fbank = gammatone_fbank(get_sr(s); sfreq=get_freq(s), kwargs...)
    ErbSpec(s, fbank; win_norm)
end

Base.eltype(::ErbSpec{F,B,T}) where {F,B,T} = T
@inline get_data(e::ErbSpec)      = e.spec'
@inline get_spec(e::ErbSpec)      = e.spec
@inline get_freq(e::ErbSpec)      = get_freq(e.fbank)
@inline get_freqrange(e::ErbSpec) = get_freqrange(e.fbank)
@inline get_setup(e::ErbSpec)     = e.info
@inline get_sr(e::ErbSpec)        = e.info.sr
@inline get_nbands(e::ErbSpec)    = get_nbands(e.fbank)
@inline get_fbank(e::ErbSpec)     = e.fbank
@inline get_parent(e::ErbSpec)    = e.parent

function Base.show(io::IO, e::ErbSpec{F,B,T}) where {F,B,T}
    nbands, nframes = size(e.spec)
    print(io, "ErbSpec{$(nameof(F)),$T}($nframes frames × $nbands bands, sr=$(get_sr(e)) Hz)")
end

function Base.show(io::IO, ::MIME"text/plain", e::ErbSpec{F,B,T}) where {F,B,T}
    nbands, nframes = size(e.spec)
    freq_range = extrema(get_freq(e))
    println(io, "ErbSpec{$(nameof(F)),$T}")
    println(io, "  Dimensions:")
    println(io, "    Frames:     $nframes")
    println(io, "    ERB bands:  $nbands")
    println(io, "  Configuration:")
    println(io, "    Sample rate:       $(get_sr(e)) Hz")
    println(io, "    Frequency range:   $(round(freq_range[1], digits=1)) - $(round(freq_range[2], digits=1)) Hz")
    print(io,   "    Window normalized: $(e.info.win_norm)")
end
