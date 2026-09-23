# ---------------------------------------------------------------------------- #
#                         the time-frequency interface                         #
# ---------------------------------------------------------------------------- #
# Every stage that consumes a spectrogram is written against the accessors in
# this file, never against a concrete front end. See docs/src/design.md.

"""
    get_spec(s::AbstractSpectrogram) -> AbstractMatrix

Return the spectrogram matrix in the canonical internal orientation:
**bins × frames** (one frame per column), real and non-negative.

This is the accessor every downstream stage uses. `get_data` is the public
view and keeps the historical orientation of each type (`Stft` returns
bins × frames, filterbank spectrograms and cepstra return frames × bands).
A new front end must implement `get_spec`.
"""
function get_spec end

"""
    get_spectrum(s::AbstractSpectrogram) -> Function

Return the spectrum kind of a front end: [`power`](@ref) (`|X|²`) or
[`magnitude`](@ref) (`|X|`).
"""
function get_spectrum end

"""
    get_complex(s::AbstractSpectrogram) -> Matrix{Complex}

The complex time-frequency coefficients of a front end, `bins × frames`,
from which `get_spec` was derived (`|X|²` or `|X|`). Front ends that can
provide it (`Stft`, `Cqt`, ...) either return the matrix they kept
(`keep_complex=true` at construction) or recompute it on request from their
`Frames`, so the memory footprint of a stage that never asks for it is
unchanged. See the [design page](@ref design_complex).
"""
function get_complex end

"""
    get_phase(s::AbstractSpectrogram) -> Matrix

Phase `atan(imag, real)` of [`get_complex`](@ref), `bins × frames`
(audioFlux `get_phase`).
"""
get_phase(s::AbstractSpectrogram) = angle.(get_complex(s))

"""
    get_window(s::AbstractSpectrogram) -> Union{AbstractVector, Nothing}

Return the time-domain analysis window of a front end, or `nothing` when the
front end has no window (a scalogram, for instance). Defaults to `nothing`.
"""
get_window(::AbstractSpectrogram) = nothing

"""
    get_winnorm(s::AbstractSpectrogram) -> Real

Scalar window-normalisation factor used when a filterbank stage is built with
`win_norm=true`. For a windowed front end it is `1 / sum(w)^2` for a power
spectrum and `1 / sum(w)` for a magnitude spectrum (MATLAB's
`WindowNormalization`). A front end without a window returns `1`, which makes
`win_norm` a safe no-op.
"""
function get_winnorm(s::AbstractSpectrogram)
    T = eltype(s)
    w = get_window(s)
    isnothing(w) && return one(T)
    return get_spectrum(s) === magnitude ? winmagnitude(one(T), w) : winpower(one(T), w)
end

"""
    get_nbins(s::AbstractSpectrogram) -> Int

Number of frequency bins (rows of `get_spec`).
"""
get_nbins(s::AbstractSpectrogram) = size(get_spec(s), 1)

"""
    get_nframes(s::AbstractAudioSpectrum) -> Int

Number of analysis frames.
"""
get_nframes(s::AbstractSpectrogram) = size(get_spec(s), 2)

"""
    get_times(s) -> AbstractVector

Centre time of every frame, in seconds. Frame `i` starts at sample
`offset + (i-1)*step` of the original signal, where `offset` is `0` unless the
frames were centre-padded, so the centre is `(offset + (i-1)*step + winsize/2) / sr`.
"""
function get_times(s::AbstractAudioSpectrum)
    T = eltype(s)
    n = get_nframes(s)
    sr = T(get_sr(s))
    return ((0:n-1) .* T(get_step(s)) .+ (T(get_offset(s)) + T(get_winsize(s)) / 2)) ./ sr
end

"""
    get_duration(s) -> Float64

Duration in seconds covered by the frames of `s`.
"""
get_duration(s::AbstractAudioSpectrum) =
    (get_offset(s) + (get_nframes(s) - 1) * get_step(s) + get_winsize(s)) / get_sr(s)

# generic helpers -------------------------------------------------------------

# indices of the bins whose frequency lies inside `freqrange` (inclusive).
# a tiny tolerance protects exact boundaries against floating-point noise on
# non-uniform grids; `Stft` overrides this with exact integer arithmetic.
function _freq_indices(freq::AbstractVector{T}, freqrange::FreqRange) where {T<:Real}
    lo, hi = T(get_low(freqrange)), T(get_hi(freqrange))
    tol = 4 * eps(T) * max(abs(hi), one(T))
    i1 = findfirst(f -> f >= lo - tol, freq)
    i2 = findlast(f -> f <= hi + tol, freq)
    (isnothing(i1) || isnothing(i2) || i2 < i1) &&
        throw(ArgumentError("No frequency bins inside freqrange = $freqrange."))
    return i1:i2
end
_freq_indices(s::AbstractSpectrogram, freqrange::FreqRange) = _freq_indices(get_freq(s), freqrange)

# every spectrogram of the pipeline is bins × frames internally
function _check_bins(s::AbstractSpectrogram, fb::AbstractFBank)
    nb = size(get_data(fb), 2)
    nf = get_nbins(s)
    nb == nf || throw(DimensionMismatch(
        "Filterbank has $nb frequency points but the spectrogram has $nf bins. " *
        "Design the filterbank on the spectrogram (`auditory_fbank(spec; ...)`)."))
    return nothing
end

# ---------------------------------------------------------------------------- #
#                          walking up the pipeline                             #
# ---------------------------------------------------------------------------- #
"""
    get_parent(s) -> stage or nothing

The stage `s` was built from. Front ends return their [`Frames`](@ref);
a stage built from a raw matrix returns `nothing`.
"""
get_parent(::AbstractAudioSpectrum) = nothing

_noparent(what) = throw(ArgumentError(
    "cannot determine $what: this stage was built from a raw matrix and has no parent stage"))

# time grid, spectrum kind and frame energy are inherited from the parent
function get_step(s::AbstractAudioSpectrum)
    p = get_parent(s); isnothing(p) && _noparent("the frame step"); get_step(p)
end
function get_winsize(s::AbstractAudioSpectrum)
    p = get_parent(s); isnothing(p) && _noparent("the frame size"); get_winsize(p)
end
"""
    get_offset(s) -> Int

Index offset (in samples, relative to the original signal) at which the first
frame starts. `0` for plain framing, negative for centre-padded frames.
"""
function get_offset(s::AbstractAudioSpectrum)
    p = get_parent(s); isnothing(p) && return 0; get_offset(p)
end
function get_spectrum(s::AbstractAudioSpectrum)
    p = get_parent(s); isnothing(p) && _noparent("the spectrum kind"); get_spectrum(p)
end
function get_energy(s::AbstractAudioSpectrum)
    p = get_parent(s); isnothing(p) && _noparent("the frame energy"); get_energy(p)
end

"""
    get_frames(s) -> Frames

The [`Frames`](@ref) at the root of the pipeline that produced `s`.
"""
function get_frames(s::AbstractAudioSpectrum)
    p = get_parent(s); isnothing(p) && _noparent("the frames"); get_frames(p)
end
get_frames(f::AbstractFrame) = f
get_nframes(s::AbstractAudioSpectrum) = size(get_data(s), 1)

get_overlap(s::AbstractAudioSpectrum) = get_winsize(s) - get_step(s)

"""
    get_frontend(s) -> AbstractSpectrogram

The time-frequency front end (`Stft`, `Cwt`, ...) at the root of the
spectral pipeline that produced `s`: the first stage whose parent is a
[`Frames`](@ref).
"""
function get_frontend(s::AbstractAudioSpectrum)
    p = get_parent(s)
    isnothing(p) && _noparent("the front end")
    return p isa AbstractFrame ? s : get_frontend(p)
end
