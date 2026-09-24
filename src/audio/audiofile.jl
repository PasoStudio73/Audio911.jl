# ---------------------------------------------------------------------------------------- #
#                                       audio utils                                        #
# ---------------------------------------------------------------------------------------- #
"""
    to_mono(x::Matrix) -> Matrix

Average the channels (columns) of a `frames × channels` signal into a
`frames × 1` matrix (librosa `to_mono`).
"""
to_mono(x::Matrix{T}) where {T<:AbstractFloat} = size(x, 2) === 1 ?
    Matrix{T}(x) : mean(x, dims=2)

"""
    normalize_peak(x) -> Array

Scale a signal so that its largest absolute sample is 1 (`x ./ maximum(abs, x)`).
A silent signal is returned unchanged.
"""
function normalize_peak(x::Array{T}) where {T<:AbstractFloat}
    m = maximum(abs, x; init=zero(T))
    return m > 0 ? x ./ m : copy(x)
end

"""
    resample(x, sr, new_sr; method=:polyphase, quality=:best, nzeros=nothing, rolloff=nothing,
             window=kaiser, beta=nothing, scale=false) -> Array

Resample a `frames × channels` signal from `sr` to `new_sr` Hz. The element
type is kept.

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
    x::Array{T},
    sr::Int,
    new_sr::Int;
    method::Symbol=:polyphase,
    scale::Bool=false,
    kwargs...
) where {T<:AbstractFloat}
    sr == new_sr && return x
    sr > 0 && new_sr > 0 ||
        throw(ArgumentError("sample rates must be positive, got $sr and $new_sr"))
    if method === :sinc
        ratio = new_sr / sr
        x isa AbstractVector && return T.(_resample_sinc(x, ratio; scale, kwargs...))
        return T.(reduce(hcat, [_resample_sinc(c, ratio; scale, kwargs...) for c in eachcol(x)]))
    end
    method === :polyphase ||
        throw(ArgumentError("method must be :polyphase or :sinc, got :$method"))
    (isempty(kwargs) && !scale) ||
        throw(ArgumentError("quality, nzeros, rolloff, window, beta and scale " *
            "only apply to method=:sinc"))
    y = DSP.resample(x, Rational(new_sr, sr); dims=1)
    return eltype(y) === T ? y : T.(y)
end

# ---------------------------------------------------------------------------------------- #
#                                    AudioFile struct                                      #
# ---------------------------------------------------------------------------------------- #
"""
    AudioFile{T} <: AbstractAudioFile

An audio signal with its sample rate, as loaded by [`load`](@ref) or wrapped
from an in-memory array with [`AudioFile(x, sr)`](@ref AudioFile(::AbstractVecOrMat, ::Int)).

The samples are stored as a `frames × channels` matrix of `T`
(`Float32` or `Float64`); a mono signal is `frames × 1`.

Accessors: [`get_data`](@ref), [`get_sr`](@ref), [`get_origin_sr`](@ref),
[`get_nchannels`](@ref), [`is_norm`](@ref), [`get_path`](@ref),
[`get_duration`](@ref), `length`, `eltype`.
"""
struct AudioFile{T<:AudioData} <: AbstractAudioFile
    data::Matrix{T}
    sr::Int
    origin_sr::Int
    norm::Bool
    path::String
end

"""
    AudioFile(x::AbstractVecOrMat, sr::Int; mono=true, norm=false, new_sr=nothing, format=eltype(x)) -> AudioFile

Wrap an in-memory signal (a vector, or a `frames × channels` matrix) sampled
at `sr` Hz. This is the entry point for audio held in matrices or data-frame
columns: wrap each column with its sample rate and feed it to the pipeline.

- `mono`: average the channels
- `norm`: peak-normalise to 1
- `new_sr`: resample to this rate
- `format`: `Float32` or `Float64` (integers default to `Float32`)
"""
function AudioFile(
    x::AbstractVecOrMat{<:AbstractFloat},
    sr::Int;
    mono::Bool=true,
    norm::Bool=false,
    new_sr::Maybe{Int}=nothing,
    format::Type=eltype(x) <: AudioData ? eltype(x) : Float32,
    path::String=""
)
    format <: AudioData ||
        throw(ArgumentError("format must be Float32 or Float64, got $format"))
    sr > 0 || throw(ArgumentError("sample rate must be positive, got $sr"))
    data = x isa AbstractVector ? reshape(x, :, 1) : x
    data = eltype(data) === format ? Matrix{format}(data) : Matrix{format}(format.(data))
    mono && size(data, 2) > 1 && (data = to_mono(data))
    target = isnothing(new_sr) ? sr : new_sr
    target == sr || (data = resample(data, sr, target))
    norm && (data = normalize_peak(data))
    return AudioFile{format}(data, target, sr, norm, String(path))
end

# ---------------------------------------------------------------------------------------- #
#                                          methods                                         #
# ---------------------------------------------------------------------------------------- #
Base.eltype(::AudioFile{T}) where T = T
Base.length(f::AudioFile) = size(f.data, 1)

"""
    get_data(a::AudioFile) -> Matrix

The samples, `frames × channels`.
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
    get_nchannels(a::AudioFile) -> Int
"""
@inline get_nchannels(a::AudioFile) = size(a.data, 2)

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
    println(io, "  Channels:    $(get_nchannels(a))")
    println(io, "  Sample rate: $(a.sr) Hz" * (a.sr == a.origin_sr ?
        "" : " (resampled from $(a.origin_sr) Hz)"))
    println(io, "  Duration:    $(round(get_duration(a), digits=3)) s")
    print(io,   "  Normalized:  $(a.norm)")
end

# ---------------------------------------------------------------------------------------- #
#                                           load                                           #
# ---------------------------------------------------------------------------------------- #
_read_audio(::Type{T}, f::File{format"MP3"}) where T = _read_mp3(T, filename(f))
_read_audio(::Type{T}, f::File) where T = _read_sndfile(T, filename(f))

"""
    load(path::String; sr=nothing, mono=true, norm=false, format=Float32) -> AudioFile
    load(file::File; kwargs...) -> AudioFile

Load a WAV, FLAC, OGG (Vorbis) or MP3 file. The format is taken from the
extension and verified against the file's first bytes; WAV, FLAC and OGG are
decoded with libsndfile, MP3 with mpg123 (16-bit PCM scaled by `1/32768`,
matching MATLAB's `audioread`).

# Keyword Arguments
- `sr::Union{Nothing,Int}=nothing`: resample to this rate (`nothing` keeps the file's rate)
- `mono::Bool=true`: average the channels
- `norm::Bool=false`: peak-normalise the samples to 1
- `format::Type=Float32`: element type, `Float32` or `Float64`

Processing order: format conversion, mono, resampling, normalisation.

# Examples
```julia
audio = load("speech.wav")                                # Float32, original rate
audio = load("speech.wav"; sr=16000, format=Float64)      # resampled, Float64
audio = load("music.mp3"; mono=false, norm=true)          # keep the channels
get_data(audio), get_sr(audio), get_nchannels(audio)
```
"""
function load(
    file::File;
    sr::Maybe{Int}=nothing,
    mono::Bool=true,
    norm::Bool=false,
    format::Type=Float32
)
    format <: AudioData ||
        throw(ArgumentError("format must be Float32 or Float64, got $format"))
    data, origin_sr = _read_audio(format, file)
    return AudioFile(data, origin_sr; mono, norm, new_sr=sr, format, path=filename(file))
end

function load(path::String; kwargs...)
    sym = detect_format(path)
    return load(File{AbstractDataFormat{sym}}(String(path)); kwargs...)
end
