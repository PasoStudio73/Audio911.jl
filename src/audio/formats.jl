# ---------------------------------------------------------------------------- #
#                               file format types                              #
# ---------------------------------------------------------------------------- #
"""
    AbstractDataFormat{sym}

Type-level tag of an audio container format (`:WAV`, `:FLAC`, `:OGG`, `:MP3`).
Build one with the [`@format_str`](@ref) string macro: `format"WAV"`.
"""
abstract type AbstractDataFormat{sym} end

"""
    @format_str(s)

`format"WAV"` is `AbstractDataFormat{:WAV}`; likewise `format"MP3"`,
`format"FLAC"` and `format"OGG"`.
"""
macro format_str(s)
    :(AbstractDataFormat{$(Expr(:quote, Symbol(s)))})
end

"""
    File{F<:AbstractDataFormat}

A path together with its audio format, `File{format"WAV"}("speech.wav")`.
[`load`](@ref) builds one after checking the extension and the file's magic
bytes; you can also construct one directly to skip the detection.
"""
struct File{F<:AbstractDataFormat}
    filename::String
    File{F}(file::AbstractString) where {F<:AbstractDataFormat} = new{F}(String(file))
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

The format tag, `:WAV`, `:FLAC`, `:OGG` or `:MP3`.
"""
formatname(::File{AbstractDataFormat{S}}) where S = S

# ---------------------------------------------------------------------------- #
#                                 magic bytes                                  #
# ---------------------------------------------------------------------------- #
const EXT2SYM = Dict{String,Symbol}(
    ".wav"  => :WAV,
    ".flac" => :FLAC,
    ".ogg"  => :OGG,
    ".mp3"  => :MP3,
)

const SUPPORTED_FORMATS = (:WAV, :FLAC, :OGG, :MP3)

_starts_with(buf::Vector{UInt8}, magic) =
    length(buf) ≥ length(magic) && all(i -> buf[i] == magic[i], eachindex(magic))

# WAV: "RIFF" .... "WAVE" (also RF64 / "RIFX" big endian are accepted by libsndfile)
_is_wav(buf)  = (_starts_with(buf, b"RIFF") || _starts_with(buf, b"RF64") || _starts_with(buf, b"RIFX")) &&
                length(buf) ≥ 12 && buf[9:12] == b"WAVE"
_is_flac(buf) = _starts_with(buf, b"fLaC")
_is_ogg(buf)  = _starts_with(buf, b"OggS")
# MP3: ID3v2 tag or an MPEG audio frame sync (11 set bits) at the start
function _is_mp3(buf)
    _starts_with(buf, b"ID3") && return true
    length(buf) ≥ 2 || return false
    return buf[1] == 0xff && (buf[2] & 0xe0) == 0xe0 && (buf[2] & 0x18) != 0x08 && (buf[2] & 0x06) != 0x00
end

const MAGIC = Dict{Symbol,Function}(:WAV => _is_wav, :FLAC => _is_flac, :OGG => _is_ogg, :MP3 => _is_mp3)

"""
    detect_format(path) -> Symbol

Format of an audio file from its extension, verified against the file's
first bytes. Throws an `ArgumentError` for an unsupported extension or when
the content does not match the extension.
"""
function detect_format(path::AbstractString)
    isfile(path) || throw(ArgumentError("File '$path' does not exist."))
    _, ext = splitext(path)
    key = lowercase(ext)
    haskey(EXT2SYM, key) || throw(ArgumentError(
        "Unsupported file format '$ext'. Supported formats: " *
        join(sort(collect(keys(EXT2SYM))), ", ")))
    sym = EXT2SYM[key]
    buf = open(io -> read(io, 12), path)
    MAGIC[sym](buf) || throw(ArgumentError(
        "File '$path' has extension '$ext' but does not appear to be a valid $(sym) file."))
    return sym
end
