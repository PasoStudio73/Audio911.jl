# ---------------------------------------------------------------------------------------- #
#                                       audio utils                                        #
# ---------------------------------------------------------------------------------------- #
"""
    to_mono(x::Matrix) -> Matrix

Average the channels (columns) of a `frames × channels` signal into a
single-channel matrix (librosa `to_mono`). Used at load time: all other
functions in this module work on mono `Vector` signals.
"""
to_mono(data::Array{T}) where {T<:AbstractFloat} = mean(data, dims=2)

"""
    normalize_peak(x::Vector) -> Vector

Scale a mono signal so that its largest absolute sample is 1
(`x ./ maximum(abs, x)`). A silent signal is returned unchanged.
"""
function normalize_peak(data::Vector{T}) where {T<:AbstractFloat}
    m = maximum(abs, data; init=zero(T))
    return m > zero(T) ? data ./ m : data
end

"""
    resample(x::Vector, sr, new_sr; method=:polyphase, quality=:best, nzeros=nothing,
             rolloff=nothing, window=kaiser, beta=nothing, scale=false) -> Vector

Resample a mono signal from `sr` to `new_sr` Hz. The element type is kept.

- `method=:polyphase`: polyphase FIR with the rational ratio `new_sr // sr`
  (DSP.jl).
- `method=:sinc`: band-limited windowed-sinc interpolation (Smith; audioFlux
  `Resample` and `WindowResample`, resampy). `quality` picks audioFlux's
  presets, `:best` (64 zero crossings, Kaiser β 14.77, roll-off 0.948),
  `:mid` (32, 11.66, 0.899) or `:fast` (16, 8.56, 0.85); `nzeros`,
  `rolloff`, `window` (a symmetric window, see [`Frames`](@ref)) and `beta`
  (the Kaiser β) override them. The output has `floor(n · new_sr / sr)`
  samples; `scale=true` divides it by `√(new_sr / sr)` (audioFlux `is_scale`).
"""
function resample(
    data::Vector{T},
    sr::Int,
    new_sr::Int;
    method::Symbol=:polyphase,
    scale::Bool=false,
    kwargs...
)::Vector{T} where {T<:AbstractFloat}
    sr == new_sr && return data
    sr > 0 && new_sr > 0 ||
        throw(ArgumentError("sample rates must be positive, got $sr and $new_sr"))
    return if method === :sinc
        ratio = new_sr / sr
        _resample_sinc(data, ratio; scale, kwargs...)
    elseif method === :polyphase
        (isempty(kwargs) && !scale) ||
            throw(ArgumentError("quality, nzeros, rolloff, window, beta and scale " *
                "only apply to method=:sinc"))
        DSP.resample(data, Rational(new_sr, sr); dims=1)
    else
        throw(ArgumentError("method must be :polyphase or :sinc, got :$method"))
    end
end

# ---------------------------------------------------------------------------------------- #
#                                    AudioFile struct                                      #
# ---------------------------------------------------------------------------------------- #
"""
    AudioFile{T} <: AbstractAudioFile

A mono audio signal with its sample rate, as loaded by [`load`](@ref) or
wrapped from an in-memory array with
[`AudioFile(x, sr)`](@ref AudioFile(::AbstractVector, ::Int)).

The samples are stored as a `Vector` of `T` (`Float32` or `Float64`);
multi-channel sources are averaged to mono at load time.

Accessors: [`get_data`](@ref), [`get_sr`](@ref), [`get_origin_sr`](@ref),
[`is_norm`](@ref), [`get_path`](@ref), [`get_duration`](@ref),
`length`, `eltype`.
"""
struct AudioFile{T<:AbstractFloat} <: AbstractAudioFile
    data::Vector{T}
    sr::Int
    origin_sr::Int
    norm::Bool
    path::String
end

"""
    AudioFile(x::AbstractVector, sr::Int; norm=false, new_sr=0,
              format=eltype(x)) -> AudioFile

Wrap an in-memory mono signal (a vector of samples) sampled at `sr` Hz.
This is the entry point for audio held in arrays or data-frame columns:
wrap each column with its sample rate and feed it to the pipeline.

- `norm`: peak-normalise to 1
- `new_sr`: resample to this rate (`0` keeps `sr`)
- `format`: `Float32` or `Float64` (integers default to `Float32`)
"""
function AudioFile(
    data::Vector{T},
    sr::Int;
    norm::Bool=false,
    new_sr::Int=0,
    path::String=""
) where {T<:AbstractFloat}
    sr > 0 || throw(ArgumentError("sample rate must be positive, got $sr"))
    target = iszero(new_sr) ? sr : new_sr
    target === sr || (data = resample(data, sr, target))
    norm && (data = normalize_peak(data))
    return AudioFile{T}(data, target, sr, norm, path)
end

# ---------------------------------------------------------------------------------------- #
#                                          methods                                         #
# ---------------------------------------------------------------------------------------- #
Base.eltype(::AudioFile{T}) where T = T
Base.length(f::AudioFile) = length(f.data)

"""
    get_data(a::AudioFile) -> Vector

The mono samples.
"""
@inline get_data(a::AudioFile) = a.data

"""
    get_sr(a::AudioFile) -> Int

Sample rate in Hz (after resampling, if any).
"""
@inline get_sr(a::AudioFile) = a.sr

"""
    get_origin_sr(a::AudioFile) -> Int

Sample rate of the source before any resampling.
"""
@inline get_origin_sr(a::AudioFile) = a.origin_sr

"""
    is_norm(a::AudioFile) -> Bool

Whether the samples were peak-normalised on load.
"""
@inline is_norm(a::AudioFile) = a.norm

"""
    get_path(a::AudioFile) -> String

Path of the source file (empty for in-memory audio).
"""
@inline get_path(a::AudioFile) = a.path

"""
    get_duration(a::AudioFile) -> Float64

Duration in seconds.
"""
get_duration(a::AudioFile) = length(a) / a.sr

function Base.show(io::IO, a::AudioFile{T}) where T
    print(io, "AudioFile{$T}($(length(a)) samples × $(get_nchannels(a)) ch, sr=$(a.sr) Hz)")
end

function Base.show(io::IO, ::MIME"text/plain", a::AudioFile{T}) where T
    println(io, "AudioFile{$T}")
    isempty(a.path) || println(io, "  Path:        $(a.path)")
    println(io, "  Samples:     $(length(a))")
    println(io, "  Sample rate: $(a.sr) Hz" * (a.sr == a.origin_sr ?
        "" : " (resampled from $(a.origin_sr) Hz)"))
    println(io, "  Duration:    $(round(get_duration(a), digits=3)) s")
    print(io,   "  Normalized:  $(a.norm)")
end

# ---------------------------------------------------------------------------------------- #
#                                           load                                           #
# ---------------------------------------------------------------------------------------- #
_read_audio(::Type{T}, f::File{Mp3}) where T = _read_mp3(T, filename(f))
_read_audio(::Type{T}, f::File{S}) where {T,S} = _read_sndfile(T, filename(f))

"""
    load(path::String; sr=0, norm=false, format=Float32) -> AudioFile
    load(file::File; kwargs...) -> AudioFile

Load a WAV, FLAC, OGG (Vorbis) or MP3 file as a mono signal (multi-channel
files are averaged to mono). The format is taken from the extension and
verified against the file's first bytes; WAV, FLAC and OGG are decoded with
libsndfile, MP3 with mpg123 (16-bit PCM scaled by `1/32768`, matching
MATLAB's `audioread`).

# Keyword Arguments
- `sr::Int=0`: resample to this rate (`0` keeps the file's rate)
- `norm::Bool=false`: peak-normalise the samples to 1
- `format::Type=Float32`: element type, `Float32` or `Float64`

Processing order: format conversion, mono mixdown, resampling, normalisation.

# Examples
```julia
audio = load("speech.wav")                                # Float32, original rate
audio = load("speech.wav"; sr=16000, format=Float64)      # resampled, Float64
audio = load("music.mp3"; norm=true)                      # peak-normalised
get_data(audio), get_sr(audio)
```
"""
function load(
    file::File{S};
    sr::Int=0,
    norm::Bool=false,
    format::Type{T}=Float32
)::AudioFile{T} where {S<:AbstractDataFormat,T<:AbstractFloat}
    format <: AbstractFloat ||
        throw(ArgumentError("format must be Float32 or Float64, got $format"))
    data, origin_sr = _read_audio(format, file)
    return AudioFile(data, origin_sr; norm, new_sr=sr, path=filename(file))
end

function load(path::String; kwargs...)
    sym = detect_format(path)
    return load(File{sym}(String(path)); kwargs...)
end

# ---------------------------------------------------------------------------------------- #
#                                           save                                           #
# ---------------------------------------------------------------------------------------- #
"""
    save(path::String, x::AbstractVector, sr::Int)
    save(path::String, a::AudioFile)

Write a mono signal to a WAV file with libsndfile. `x` is a vector of
samples in `[-1, 1]`.

# Examples
```julia
save("out.wav", audio)
save("out.wav", get_data(audio), get_sr(audio))
```
"""
function save(path::String, data::Vector{T}, sr::Int) where {T<:AbstractFloat}
    sr > 0 || throw(ArgumentError("sample rate must be positive, got $sr"))
    _write_sndfile(String(path), data, sr)
    return path
end

save(path::String, a::AudioFile{T}) where T = save(path, get_data(a), get_sr(a))