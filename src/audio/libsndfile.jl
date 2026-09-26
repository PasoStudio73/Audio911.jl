# ---------------------------------------------------------------------------------------- #
#                                    libsndfile bindings                                   #
# ---------------------------------------------------------------------------------------- #
# Minimal bindings to libsndfile (WAV, FLAC, OGG/Vorbis). Multi-channel files
# are averaged to mono on read; the whole Audio911 pipeline works exclusively
# on mono Vector signals. Writing outputs mono 16-bit PCM WAV.
const SFM_READ = Int32(0x10)
const SFM_WRITE = Int32(0x20)

const SF_FORMAT_WAV = Int32(0x010000)
const SF_FORMAT_PCM_16 = Int32(0x0002)

mutable struct SF_INFO
    frames::Int
    samplerate::Int32
    channels::Int32
    format::Int32
    sections::Int32
    seekable::Int32
end
SF_INFO() = SF_INFO(0, 0, 0, 0, 0, 0)

function _sf_strerror(ptr::Ptr{Cvoid})
    s = ccall((:sf_strerror, libsndfile), Ptr{Cchar}, (Ptr{Cvoid},), ptr)
    return s === C_NULL ? "unknown libsndfile error" : unsafe_string(s)
end

function _sf_open(path::String, info::SF_INFO)
    ptr = ccall(
        (:sf_open, libsndfile),
        Ptr{Cvoid},
        (Cstring, Int32, Ref{SF_INFO}),
        path,
        SFM_READ,
        info
    )
    ptr === C_NULL && throw(ArgumentError(
        "libsndfile could not open '$path': $(_sf_strerror(C_NULL))"))
    return ptr
end

_sf_close(ptr::Ptr{Cvoid}) = ccall((:sf_close, libsndfile), Int32, (Ptr{Cvoid},), ptr)

_sf_readf(ptr, dest::Matrix{Float32}, n) = ccall(
    (:sf_readf_float, libsndfile), Int, (Ptr{Cvoid}, Ptr{Float32}, Int), ptr, dest, n)
_sf_readf(ptr, dest::Matrix{Float64}, n) = ccall(
    (:sf_readf_double, libsndfile), Int, (Ptr{Cvoid}, Ptr{Float64}, Int), ptr, dest, n)

# read a whole file as a mono Vector{T}; samples are scaled to [-1, 1).
# Multi-channel files are averaged to mono (see `to_mono`): the whole Audio911
# analysing pipeline is intended to work exclusively on mono signals.
function _read_sndfile(::Type{T}, path::String) where {T<:AudioData}
    info = SF_INFO()
    ptr  = _sf_open(path, info)
    try
        nch = Int(info.channels)
        nfr = Int(info.frames)
        # libsndfile delivers interleaved frames: (channels × frames) column-major
        buf = Matrix{T}(undef, nch, max(nfr, 0))
        got = nfr > 0 ? Int(_sf_readf(ptr, buf, nfr)) : 0
        data = Matrix{T}(undef, got, nch)
        @inbounds for c in 1:nch, i in 1:got
            data[i, c] = buf[c, i]
        end
        # mono mixdown: average channels, then continue with a Vector
        size(data, 2) > 1 && (data = to_mono(data))
        return vec(data), Int(info.samplerate)
    finally
        _sf_close(ptr)
    end
end

_sf_writef(ptr, src::Vector{Float32}) = ccall(
    (:sf_writef_float, libsndfile),
    Int,
    (Ptr{Cvoid}, Ptr{Float32}, Int),
    ptr,
    src,
    length(src)
)
_sf_writef(ptr, src::Vector{Float64}) = ccall(
    (:sf_writef_double, libsndfile),
    Int,
    (Ptr{Cvoid}, Ptr{Float64}, Int),
    ptr,
    src,
    length(src)
)

# write a mono signal (Vector of samples in [-1, 1]) to a 16-bit PCM WAV file at sr Hz
function _write_sndfile(path::String, data::Vector{T}, sr::Int) where {T<:AbstractFloat}
    info = SF_INFO(0, sr, 1, SF_FORMAT_WAV | SF_FORMAT_PCM_16, 0, 0)
    ptr = ccall((:sf_open, libsndfile), Ptr{Cvoid},
        (Cstring, Int32, Ref{SF_INFO}), path, SFM_WRITE, info)
    ptr === C_NULL && error(
        "libsndfile could not open '$path' for writing: $(_sf_strerror(C_NULL))")
    try
        # libsndfile expects interleaved frames: (channels × frames) column-major
        n = Int(_sf_writef(ptr, data))
        n === length(data) || error("libsndfile: wrote $n of $(size(data, 1)) frames")
    finally
        _sf_close(ptr)
    end
    return nothing
end
