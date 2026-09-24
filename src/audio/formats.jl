# ---------------------------------------------------------------------------------------- #
#                                     file format types                                    #
# ---------------------------------------------------------------------------------------- #
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

# ---------------------------------------------------------------------------------------- #
#                                       magic bytes                                        #
# ---------------------------------------------------------------------------------------- #
function evalext(e::UInt8)
    e == 0x01 ? :WAV :
    e == 0x02 ? :FLAC :
    e == 0x03 ? :OGG : :MP3
end

const Formats = Dict{String,UInt8}(
    ".wav" => 0x01,
    ".flac" => 0x02,
    ".ogg" => 0x03,
    ".mp3" => 0x04,
)

function magic(e::UInt8)
    e == 0x01 ? _is_wav :
    e == 0x02 ? _is_flac :
    e == 0x03 ? _is_ogg : _is_mp3
end

_starts_with(buf::Vector{UInt8}, magic) =
    length(buf) ≥ length(magic) && all(i -> buf[i] == magic[i], eachindex(magic))

function magic(buf::Vector{UInt8}, ext::UInt8)
    # WAV: "RIFF" .... "WAVE" (also RF64 / "RIFX" big endian are accepted by libsndfile)
    return if ext === 0x01
        (_starts_with(buf, b"RIFF") ||
            _starts_with(buf, b"RF64") ||
            _starts_with(buf, b"RIFX")
        ) && length(buf) ≥ 12 && buf[9:12] == b"WAVE"
    elseif ext === 0x02
        _starts_with(buf, b"fLaC")
    elseif ext === 0x03
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
function detect_format(path::AbstractString)
    isfile(path) || throw(ArgumentError("File '$path' does not exist."))
    _, ext = splitext(path)
    key = lowercase(ext)
    haskey(Formats, key) || throw(ArgumentError(
        "Unsupported file format '$ext'. Supported formats: " *
        join(sort(collect(keys(Formats))), ", ")))
    ext = Formats[key]
    buf = open(io -> read(io, 12), path)
    magic(buf, ext) || throw(ArgumentError(
        "File '$path' has extension '$ext' but does not appear to be a valid $(ext) file."))
    return evalext(ext)
end
