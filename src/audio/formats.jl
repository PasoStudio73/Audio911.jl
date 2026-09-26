# ---------------------------------------------------------------------------------------- #
#                                     file format types                                    #
# ---------------------------------------------------------------------------------------- #
"""
    SUPPORTED_FORMATS

Supported audio container formats, identified by `Symbol`: `:wav`, `:flac`,
`:ogg` (Vorbis) and `:mp3`. [`detect_format`](@ref) returns one of these from
a file's extension and magic bytes; [`File`](@ref) carries it as a field.
"""
const SUPPORTED_FORMATS = (:wav, :flac, :ogg, :mp3)

"""
    File

A path together with its audio format symbol, e.g. `File(:wav, "speech.wav")`.
[`load`](@ref) builds one after checking the extension and the file's magic
bytes; you can also construct one directly to skip the detection.
"""
struct File
    format::Symbol
    filename::String
    function File(format::Symbol, file::String)
        format in SUPPORTED_FORMATS || throw(ArgumentError(
            "Unsupported format ':$format'. Supported formats: $(SUPPORTED_FORMATS)"))
        new(format, file)
    end
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

The format symbol: `:wav`, `:flac`, `:ogg` or `:mp3`.
"""
formatname(f::File) = f.format

# ---------------------------------------------------------------------------------------- #
#                                       magic bytes                                        #
# ---------------------------------------------------------------------------------------- #
function evalext(e::String)
    e == ".wav" ? :wav :
    e == ".flac" ? :flac :
    e == ".ogg" ? :ogg :
    e == ".mp3" ? :mp3 :
    throw(ArgumentError("Unsupported file format '$e'. Supported formats: " *
        ".wav, .flac, .ogg, .mp3"))
end

_starts_with(buf::Vector{UInt8}, magic) =
    length(buf) ≥ length(magic) && all(i -> buf[i] == magic[i], eachindex(magic))

function magic(buf::Vector{UInt8}, fmt::Symbol)
    # WAV: "RIFF" .... "WAVE" (also RF64 / "RIFX" big endian are accepted by libsndfile)
    return if fmt === :wav
        (_starts_with(buf, b"RIFF") ||
            _starts_with(buf, b"RF64") ||
            _starts_with(buf, b"RIFX")
        ) && length(buf) ≥ 12 && buf[9:12] == b"WAVE"
    elseif fmt === :flac
        _starts_with(buf, b"fLaC")
    elseif fmt === :ogg
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

Format of an audio file from its extension (`:wav`, `:flac`, `:ogg` or
`:mp3`), verified against the file's first bytes. Throws an `ArgumentError`
for an unsupported extension or when the content does not match the extension.
"""
function detect_format(path::String)
    isfile(path) || throw(ArgumentError("File '$path' does not exist."))
    _, e = splitext(path)
    fmt = evalext(lowercase(e))
    buf = open(io -> read(io, 12), path)
    magic(buf, fmt) || throw(ArgumentError(
        "File '$path' has extension '$e' but does not appear to be a valid $(fmt) file."))
    return fmt
end