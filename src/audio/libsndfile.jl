# ---------------------------------------------------------------------------- #
#                              libsndfile bindings                             #
# ---------------------------------------------------------------------------- #
# Minimal read-only bindings to libsndfile (WAV, FLAC, OGG/Vorbis).

const SFM_READ = Int32(0x10)

mutable struct SF_INFO
    frames     :: Int64
    samplerate :: Int32
    channels   :: Int32
    format     :: Int32
    sections   :: Int32
    seekable   :: Int32
end
SF_INFO() = SF_INFO(0, 0, 0, 0, 0, 0)

function _sf_strerror(ptr::Ptr{Cvoid})
    s = ccall((:sf_strerror, libsndfile), Ptr{Cchar}, (Ptr{Cvoid},), ptr)
    return s == C_NULL ? "unknown libsndfile error" : unsafe_string(s)
end

function _sf_open(path::String, info::SF_INFO)
    ptr = ccall((:sf_open, libsndfile), Ptr{Cvoid}, (Cstring, Int32, Ref{SF_INFO}), path, SFM_READ, info)
    ptr == C_NULL && throw(ArgumentError("libsndfile could not open '$path': $(_sf_strerror(C_NULL))"))
    return ptr
end

_sf_close(ptr::Ptr{Cvoid}) = ccall((:sf_close, libsndfile), Int32, (Ptr{Cvoid},), ptr)

_sf_readf(ptr, dest::Matrix{Float32}, n) =
    ccall((:sf_readf_float, libsndfile), Int64, (Ptr{Cvoid}, Ptr{Float32}, Int64), ptr, dest, n)
_sf_readf(ptr, dest::Matrix{Float64}, n) =
    ccall((:sf_readf_double, libsndfile), Int64, (Ptr{Cvoid}, Ptr{Float64}, Int64), ptr, dest, n)

# read a whole file as (frames × channels) in T; samples are scaled to [-1, 1)
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
        return data, Int(info.samplerate)
    finally
        _sf_close(ptr)
    end
end
