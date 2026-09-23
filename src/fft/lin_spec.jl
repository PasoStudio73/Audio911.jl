# ---------------------------------------------------------------------------- #
#                                    info                                      #
# ---------------------------------------------------------------------------- #
struct LinSpecSetup <: AbstractSetup
    sr            :: Int64
    freqrange     :: FreqRange
    spectrum_type :: Base.Callable
    win_norm      :: Bool
end

# ---------------------------------------------------------------------------- #
#                          linear spectrogram struct                           #
# ---------------------------------------------------------------------------- #
"""
    LinSpec{F,T} <: AbstractSpectrogram

A front-end spectrogram restricted to a frequency range, scaled to a
one-sided spectrum and optionally window-normalised (MATLAB's
`linearSpectrum`). `F` is the type of the front end it was built from.

Build one with [`LinSpec(spec; freqrange, win_norm)`](@ref LinSpec(::AbstractSpectrogram)).
"""
struct LinSpec{F,T} <: AbstractSpectrogram
    spec   :: Matrix{T}
    freq   :: Vector{T}
    parent :: F
    info   :: LinSpecSetup
end

# ---------------------------------------------------------------------------- #
#                                    methods                                   #
# ---------------------------------------------------------------------------- #
Base.eltype(::LinSpec{F,T}) where {F,T} = T

"""
    get_data(s::LinSpec) -> AbstractMatrix

The spectrogram transposed to `frames × bins` (MATLAB orientation).
"""
get_data(s::LinSpec)  = s.spec'
get_spec(s::LinSpec)  = s.spec

"""
    get_freq(s::LinSpec) -> Vector

Bin frequencies in Hz.
"""
get_freq(s::LinSpec)  = s.freq
get_setup(s::LinSpec) = s.info
get_sr(s::LinSpec)    = s.info.sr
get_spectrum(s::LinSpec)  = s.info.spectrum_type
get_freqrange(s::LinSpec) = s.info.freqrange
get_parent(s::LinSpec)    = s.parent

# ---------------------------------------------------------------------------- #
#                                     show                                     #
# ---------------------------------------------------------------------------- #
function Base.show(io::IO, s::LinSpec{F,T}) where {F,T}
    nfreqs, nframes = size(s.spec)
    freq_range = extrema(get_freq(s))
    print(io, "LinSpec{$(nameof(F)),$T}($nframes frames × $nfreqs bins, ")
    print(io, "$(round(freq_range[1], digits=0))-$(round(freq_range[2], digits=0)) Hz, ")
    print(io, "sr=$(s.info.sr) Hz, win_norm=$(s.info.win_norm))")
end

function Base.show(io::IO, ::MIME"text/plain", s::LinSpec{F,T}) where {F,T}
    nfreqs, nframes = size(s.spec)
    freq_range = extrema(get_freq(s))
    println(io, "LinSpec{$(nameof(F)),$T}")
    println(io, "  Dimensions:")
    println(io, "    Frames:          $nframes")
    println(io, "    Frequency bins:  $nfreqs")
    println(io, "  Configuration:")
    println(io, "    Sample rate:        $(s.info.sr) Hz")
    println(io, "    Frequency range:    $(round(freq_range[1], digits=1)) - $(round(freq_range[2], digits=1)) Hz")
    println(io, "    Spectrum type:      $(s.info.spectrum_type)")
    print(io,   "    Window normalized:  $(s.info.win_norm)")
end

# ---------------------------------------------------------------------------- #
#                                 get lin spec                                 #
# ---------------------------------------------------------------------------- #
"""
    LinSpec(spec::AbstractSpectrogram; freqrange=(0, sr÷2), win_norm=false) -> LinSpec

Linear spectrum of any front end (`Stft`, `Cwt`, ...), reproducing MATLAB's
`linearSpectrum` feature.

The bins whose frequency lies in `freqrange` are kept, every bin strictly
between DC and Nyquist is doubled to account for the negative frequencies of
the one-sided spectrum, and, when `win_norm=true`, the result is divided by
`sum(w)^2` (power) or `sum(w)` (magnitude) where `w` is the analysis window
(a front end without a window is left unscaled).

# Keyword Arguments
- `freqrange::FreqRange=(0, sr÷2)`: frequency range to keep, in Hz
- `win_norm::Bool=false`: window normalisation (MATLAB `WindowNormalization`)

# Examples
```julia
stft    = Stft(audio; winsize=512, winstep=256, type=hamming)
linspec = LinSpec(stft; freqrange=(100, 8000), win_norm=true)
get_data(linspec)   # frames × bins
get_freq(linspec)   # bin frequencies
```
"""
function LinSpec(
    s         :: AbstractSpectrogram;
    freqrange :: FreqRange=(0, get_sr(s) >> 1),
    win_norm  :: Bool=false
)
    T    = eltype(s)
    sr   = get_sr(s)
    idx  = _freq_indices(s, freqrange)
    freq = Vector{T}(get_freq(s)[idx])
    spec = Matrix{T}(view(get_spec(s), idx, :))

    factor = win_norm ? get_winnorm(s) : one(T)
    nyq = T(sr) / 2
    tol = 8 * eps(T) * nyq
    @inbounds for (k, f) in enumerate(freq)
        g = (f > tol && f < nyq - tol) ? 2factor : factor
        g == one(T) && continue
        @views spec[k, :] .*= g
    end

    info = LinSpecSetup(sr, freqrange, get_spectrum(s), win_norm)
    return LinSpec{typeof(s),T}(spec, freq, s, info)
end
