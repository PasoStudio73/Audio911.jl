# ---------------------------------------------------------------------------- #
#                                    info                                      #
# ---------------------------------------------------------------------------- #
struct StftSetup{T<:AudioData} <: AbstractSetup
    sr       :: Int64
    nfft     :: Int64
    winsize  :: Int64
    winstep  :: Int64
    overlap  :: Int64
    spectrum :: Base.Callable
    window   :: Vector{T}
    offset   :: Int64
    scale    :: Float64
end

# ---------------------------------------------------------------------------- #
#                                 stft struct                                  #
# ---------------------------------------------------------------------------- #
"""
    Stft{T} <: AbstractSpectrogram

Short-time Fourier transform of a [`Frames`](@ref) object: a one-sided
`power` or `magnitude` spectrogram stored as `bins × frames`, its frequency
axis (`sr/nfft` apart) and an `info` record (sample rate, `nfft`, window,
hop). It is the default time-frequency front end of the pipeline; see
[`Cwt`](@ref) for the wavelet alternative.

Build one with [`Stft(frames; nfft, spectrum)`](@ref Stft(::Frames)) or
[`Stft(audio; kwargs...)`](@ref Stft(::AudioFile)).
"""
struct Stft{T<:AudioData} <: AbstractSpectrogram
    spec   :: Matrix{T}
    freq   :: StepRangeLen{T}
    frames :: Frames{T}
    info   :: StftSetup{T}
end

#------------------------------------------------------------------------------#
#                                   methods                                    #
#------------------------------------------------------------------------------#
Base.eltype(::Stft{T}) where T = T

"""
    get_data(s::Stft) -> Matrix

The spectrogram as stored: `bins × frames` (same as [`get_spec`](@ref)).
"""
@inline get_data(s::Stft) = s.spec
@inline get_spec(s::Stft) = s.spec

"""
    get_freq(s::Stft) -> AbstractRange

Bin centre frequencies in Hz, `0:sr/nfft:sr/2`.
"""
@inline get_freq(s::Stft) = s.freq

"""
    get_setup(s::Stft) -> StftSetup

Parameters used to compute the STFT.
"""
@inline get_setup(s::Stft) = s.info

"""
    get_sr(s::Stft) -> Int

Sample rate in Hz.
"""
@inline get_sr(s::Stft) = s.info.sr

"""
    get_nfft(s::Stft) -> Int

FFT size. The one-sided spectrum has `nfft ÷ 2 + 1` bins.
"""
@inline get_nfft(s::Stft) = s.info.nfft

"""
    get_spectrum(s::Stft) -> Function

`power` or `magnitude`.
"""
@inline get_spectrum(s::Stft) = s.info.spectrum

@inline get_winsize(s::Stft) = s.info.winsize
@inline get_step(s::Stft)    = s.info.winstep
@inline get_overlap(s::Stft) = s.info.overlap
@inline get_offset(s::Stft)  = s.info.offset

"""
    get_window(s::Stft) -> Vector

The time-domain analysis window.
"""
@inline get_window(s::Stft) = s.info.window

"""
    get_frames(s::Stft) -> Frames

The frames the transform was computed from.
"""
@inline get_frames(s::Stft) = s.frames
@inline get_parent(s::Stft) = s.frames
@inline get_energy(s::Stft) = get_energy(s.frames)

# exact bin arithmetic for the FFT grid (MATLAB parity)
function _freq_indices(s::Stft, freqrange::FreqRange)
    nfft, sr = get_nfft(s), get_sr(s)
    bin_low  = cld(get_low(freqrange) * nfft, sr) + 1
    bin_high = fld(get_hi(freqrange)  * nfft, sr) + 1
    bin_high = min(bin_high, get_nbins(s))
    bin_low ≤ bin_high || throw(ArgumentError("No frequency bins inside freqrange = $freqrange."))
    return bin_low:bin_high
end

# ---------------------------------------------------------------------------- #
#                                     show                                     #
# ---------------------------------------------------------------------------- #
function Base.show(io::IO, s::Stft{T}) where T
    nfreqs, nframes = size(get_data(s))
    print(io, "Stft{$T}($nframes frames × $nfreqs bins, sr=$(s.info.sr) Hz, spectrum=$(s.info.spectrum))")
end

function Base.show(io::IO, ::MIME"text/plain", s::Stft{T}) where T
    nfreqs, nframes = size(get_data(s))
    println(io, "Stft{$T}")
    println(io, "    Sample rate:     $(s.info.sr) Hz")
    println(io, "    Frames:          $nframes")
    println(io, "    Bins:            $nfreqs")
    println(io, "    FFT size:        $(s.info.nfft)")
    println(io, "    Window size:     $(s.info.winsize) samples")
    println(io, "    Hop size:        $(s.info.winstep) samples")
    println(io, "    Overlap:         $(s.info.overlap) samples")
    print(io,   "    Spectrum type:   $(s.info.spectrum)")
end

#------------------------------------------------------------------------------#
#                           spectrum normalizations                            #
#------------------------------------------------------------------------------#
"""
    power(f)

Power spectrum of complex FFT values: `|X(f)|²`. Works on scalars and arrays.
"""
power(f) = abs2.(f)

"""
    magnitude(f)

Magnitude spectrum of complex FFT values: `|X(f)|`. Works on scalars and arrays.
"""
magnitude(f) = abs.(f)

#------------------------------------------------------------------------------#
#                                  utilities                                   #
#------------------------------------------------------------------------------#
_onesided_length(nfft::Int64) = nfft ÷ 2 + 1

# partition 1:n into at most nthreads contiguous chunks
function _chunks(n::Int)
    nt = max(1, min(Threads.nthreads(), n))
    return collect(Iterators.partition(1:n, cld(n, nt)))
end

# stream the windowed frames through a pre-planned real FFT, one frame at a
# time, writing `spectrum(X)` straight into the output columns
function _stft!(spec::Matrix{T}, frames::Frames{T}, nfft::Int, spectrum::Base.Callable) where T
    w  = get_window(frames)
    ws = length(w)
    n  = size(spec, 2)
    n == 0 && return spec
    plan = plan_rfft(zeros(T, nfft))
    Threads.@threads for chunk in _chunks(n)
        buf = zeros(T, nfft)
        out = Vector{Complex{T}}(undef, _onesided_length(nfft))
        raw = view(buf, 1:ws)
        for j in chunk
            frame!(raw, frames, j)
            @inbounds @simd for k in 1:ws
                buf[k] *= w[k]
            end
            mul!(out, plan, buf)
            @inbounds for k in eachindex(out)
                spec[k, j] = spectrum(out[k])
            end
        end
    end
    return spec
end

#------------------------------------------------------------------------------#
#                                   get stft                                   #
#------------------------------------------------------------------------------#
"""
    Stft(frames::Frames; nfft=get_winsize(frames), spectrum=power) -> Stft

Compute the short-time Fourier transform of pre-computed frames.

Every frame is multiplied by the analysis window, zero-padded to `nfft` when
`nfft > winsize`, transformed with a pre-planned real FFT and reduced to a
one-sided `power` (`|X|²`) or `magnitude` (`|X|`) spectrum. The work streams
through the frames with one buffer per thread, so memory is the output matrix
plus a few `nfft`-length buffers.

# Keyword Arguments
- `nfft::Int`: FFT size, must be `≥ winsize` (default: `winsize`). Zero
  padding interpolates the spectrum; frequency spacing is `sr / nfft`.
- `spectrum::Base.Callable`: `power` (default) or `magnitude`
- `scale::Real=1`: multiply the spectrum by a constant (`1/nfft` reproduces
  python_speech_features' `powspec`)

# Throws
`ArgumentError` if `nfft < winsize` or the frames do not overlap.

# Examples
```julia
frames = Frames(audio; winsize=512, winstep=256, type=hamming)
stft   = Stft(frames)                       # 257 × nframes power spectrogram
stft   = Stft(frames; nfft=1024)            # finer frequency grid
stft   = Stft(frames; spectrum=magnitude)
```

See also [`Frames`](@ref), [`power`](@ref), [`magnitude`](@ref), [`Cwt`](@ref).
"""
function Stft(
    frames   :: Frames{T};
    nfft     :: Int64=get_winsize(frames),
    spectrum :: Base.Callable=power,
    scale    :: Real=1,
) where {T<:AudioData}
    sr      = get_sr(frames)
    winsize = get_winsize(frames)
    overlap = get_overlap(frames)

    (0 ≤ overlap < winsize) ||
        throw(ArgumentError("Overlap length must be < window length. " *
                "Got overlap = $overlap, window length = $winsize"))
    nfft < winsize &&
        throw(ArgumentError("nfft must be ≥ window length. " *
                "Got nfft = $nfft, window length = $winsize"))
    spectrum in (power, magnitude) ||
        throw(ArgumentError("spectrum must be `power` or `magnitude`, got $spectrum"))

    spec = Matrix{T}(undef, _onesided_length(nfft), length(frames))
    _stft!(spec, frames, nfft, spectrum)
    scale == 1 || (spec .*= T(scale))

    freq = (0:size(spec, 1)-1) .* (T(sr) / T(nfft))
    info = StftSetup{T}(sr, nfft, winsize, get_step(frames), overlap, spectrum,
                        get_window(frames), get_offset(frames), Float64(scale))

    return Stft{T}(spec, freq, frames, info)
end

"""
    Stft(audio::AudioFile; kwargs...) -> Stft
    Stft(x::AbstractVecOrMat, sr::Int; kwargs...) -> Stft

Frame the signal and compute its STFT in one call. Framing keywords
(`winsize`, `winstep`, `type`, `periodic`, `center`, `pad_mode`, `preemph`,
`dc_removal`) go to [`Frames`](@ref); `nfft` and `spectrum` go to
[`Stft(::Frames)`](@ref).

```julia
stft = Stft(audio; winsize=1024, winstep=512, type=hamming, nfft=2048, spectrum=magnitude)
```
"""
function Stft(
    audio      :: AbstractVecOrMat{<:Real},
    sr         :: Int64;
    winsize    :: Int64=sr ≤ 8000 ? 256 : 512,
    winstep    :: Int64=winsize ÷ 2,
    win        :: Maybe{NamedTuple}=nothing,
    type       :: Base.Callable=hanning,
    periodic   :: Bool=true,
    center     :: Bool=false,
    pad_mode   :: Symbol=:constant,
    preemph    :: Real=0,
    dc_removal :: Bool=false,
    pad_end    :: Bool=false,
    kwargs...
)
    frames = Frames(audio, sr; winsize, winstep, win, type, periodic, center, pad_mode, preemph, dc_removal, pad_end)
    return Stft(frames; kwargs...)
end

Stft(a::AudioFile; kwargs...) = Stft(get_data(a), get_sr(a); kwargs...)
