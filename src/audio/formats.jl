# ---------------------------------------------------------------------------------------- #
#                                     file format types                                    #
# ---------------------------------------------------------------------------------------- #
"""
    AbstractDataFormat

Abstract supertype of the audio container format tags. Each supported format
is a concrete singleton subtype: [`Wav`](@ref), [`Flac`](@ref), [`Ogg`](@ref)
(Vorbis) and [`Mp3`](@ref). [`detect_format`](@ref) returns one of these from
a file's extension and magic bytes; [`File{F}`](@ref File) carries it as a
type parameter.
"""
abstract type AbstractDataFormat end

"""WAV container format tag (see [`AbstractDataFormat`](@ref))."""
struct Wav <: AbstractDataFormat end

"""FLAC container format tag (see [`AbstractDataFormat`](@ref))."""
struct Flac <: AbstractDataFormat end

"""OGG (Vorbis) container format tag (see [`AbstractDataFormat`](@ref))."""
struct Ogg <: AbstractDataFormat end

"""MP3 container format tag (see [`AbstractDataFormat`](@ref))."""
struct Mp3 <: AbstractDataFormat end

"""
    File{F<:AbstractDataFormat}

A path together with its audio format, `File{format"WAV"}("speech.wav")`.
[`load`](@ref) builds one after checking the extension and the file's magic
bytes; you can also construct one directly to skip the detection.
"""
struct File{F<:AbstractDataFormat}
    filename::String
    File{F}(file::String) where {F<:AbstractDataFormat} = new{F}(file)
end

"""
    filename(f::File) -> String
"""
filename(f::File) = f.filename

"""
    file_extension(f::File) -> String
"""
file_extension(f::File) = splitext(f.filename)[2]

"""
    formatname(f::File) -> Symbol

The format tag, `Wav`, `Flac`, `Ogg` or `Mp3`.
"""
formatname(::File{S}) where S = S

# ---------------------------------------------------------------------------------------- #
#                                       magic bytes                                        #
# ---------------------------------------------------------------------------------------- #
function evalext(e::UInt8)
    e === 0x01 ? Wav :
    e === 0x02 ? Flac :
    e === 0x03 ? Ogg : Mp3
end

function evalext(e::String)
    e === ".wav" ? Wav :
    e === ".flac" ? Flac :
    e === ".ogg" ? Ogg :
    e === ".mp3" ? Mp3 :
    throw(ArgumentError("Unsupported file format '$e'. Supported formats: " *
        ".wav, .flac, .ogg, .mp3"))
end

function formats(e::Type{<:AbstractDataFormat})
    e === Wav ? 0x01 :
    e === Flac ? 0x02 :
    e === Ogg ? 0x03 : 0x04
end

function magic(e::UInt8)
    e === 0x01 ? _is_wav :
    e === 0x02 ? _is_flac :
    e === 0x03 ? _is_ogg : _is_mp3
end

_starts_with(buf::Vector{UInt8}, magic) =
    length(buf) ≥ length(magic) && all(i -> buf[i] == magic[i], eachindex(magic))

function magic(buf::Vector{UInt8}, ext::UInt8)
    # WAV: "RIFF" .... "WAVE" (also RF64 / "RIFX" big endian are accepted by libsndfile)
    return if evalext(ext) === Wav
        (_starts_with(buf, b"RIFF") ||
            _starts_with(buf, b"RF64") ||
            _starts_with(buf, b"RIFX")
        ) && length(buf) ≥ 12 && buf[9:12] == b"WAVE"
    elseif evalext(ext) === Flac
        _starts_with(buf, b"fLaC")
    elseif evalext(ext) === Ogg
        _starts_with(buf, b"OggS")
    else
    # MP3: ID3v2 tag or an MPEG audio frame sync (11 set bits) at the start
        _starts_with(buf, b"ID3") && return true
        length(buf) ≥ 2 || return false
        buf[1] === 0xff && (buf[2] & 0xe0) === 0xe0 &&
            (buf[2] & 0x18) != 0x08 && (buf[2] & 0x06) != 0x00
    end
end

# ---------------------------------------------------------------------------------------- #
#                                      detect format                                       #
# ---------------------------------------------------------------------------------------- #
"""
    detect_format(path) -> Symbol

Format of an audio file from its extension, verified against the file's
first bytes. Throws an `ArgumentError` for an unsupported extension or when
the content does not match the extension.
"""
function detect_format(path::String)
    isfile(path) || throw(ArgumentError("File '$path' does not exist."))
    _, e = splitext(path)
    key = evalext(lowercase(e))
    ext = formats(key)
    buf = open(io -> read(io, 12), path)
    magic(buf, ext) || throw(ArgumentError(
        "File '$path' has extension '$ext' but does not appear to be a valid $(ext) file."))
    return evalext(ext)
end
