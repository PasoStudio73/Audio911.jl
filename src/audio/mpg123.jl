# ---------------------------------------------------------------------------------------- #
#                                      mpg123 bindings                                     #
# ---------------------------------------------------------------------------------------- #
# Minimal bindings to libmpg123 for decoding MP3 files to 16-bit PCM.
# Multi-channel files are averaged to mono after decoding; the whole Audio911
# pipeline works exclusively on mono Vector signals.
const MPG123_OK = Cint(0)
const MPG123_DONE = Cint(-12)
const MPG123_NEW_FORMAT = Cint(-11)
const MPG123_NEED_MORE = Cint(-10)
const MPG123_ENC_SIGNED_16 = Cint(0x40 | 0x80 | 0x10)

const MPG123_HANDLE = Ptr{Cvoid}

function _mpg123_strerror(err::Cint)
    s = ccall((:mpg123_plain_strerror, libmpg123), Ptr{Cchar}, (Cint,), err)
    return s === C_NULL ? "unknown mpg123 error" : unsafe_string(s)
end

function _mpg123_new()
    err = Ref{Cint}(0)
    mh = ccall(
        (:mpg123_new, libmpg123),
        MPG123_HANDLE,
        (Ptr{Cchar}, Ref{Cint}),
        C_NULL,
        err
    )
    (mh === C_NULL || err[] != MPG123_OK) && throw(ArgumentError("could not create " *
        "an mpg123 handle: $(_mpg123_strerror(err[]))"))
    return mh
end

_mpg123_delete(mh) = ccall((:mpg123_delete, libmpg123), Cvoid, (MPG123_HANDLE,), mh)
_mpg123_close(mh) = ccall((:mpg123_close, libmpg123), Cint, (MPG123_HANDLE,), mh)

function _mpg123_open(mh, path::String)
    err = ccall((:mpg123_open, libmpg123), Cint, (MPG123_HANDLE, Cstring), mh, path)
    err === MPG123_OK || throw(ArgumentError("mpg123 could not open " *
        "'$path': $(_mpg123_strerror(err))"))
    return nothing
end

function _mpg123_getformat(mh)
    rate, nch, enc = Ref{Clong}(0), Ref{Cint}(0), Ref{Cint}(0)
    err = ccall((:mpg123_getformat, libmpg123), Cint,
                (MPG123_HANDLE, Ref{Clong}, Ref{Cint}, Ref{Cint}), mh, rate, nch, enc)
    err === MPG123_OK || throw(ArgumentError("mpg123 could not read " *
        "the stream format: $(_mpg123_strerror(err))"))
    return Int(rate[]), Int(nch[]), enc[]
end

# force signed 16-bit output for every rate/channel combination
function _mpg123_force_s16(mh)
    ccall((:mpg123_format_none, libmpg123), Cint, (MPG123_HANDLE,), mh)
    ccall((:mpg123_format_all, libmpg123), Cint, (MPG123_HANDLE,), mh)
    return nothing
end

_mpg123_length(mh) = Int(ccall((:mpg123_length, libmpg123), Int, (MPG123_HANDLE,), mh))
_mpg123_outblock(mh) = Int(
    ccall((:mpg123_outblock, libmpg123), Csize_t, (MPG123_HANDLE,), mh))

function _mpg123_read!(mh, buf::Vector{Int16})
    done = Ref{Csize_t}(0)
    err = ccall((:mpg123_read, libmpg123), Cint,
                (MPG123_HANDLE, Ptr{Int16}, Csize_t, Ref{Csize_t}),
                mh, buf, sizeof(buf), done)
    (err === MPG123_OK || err === MPG123_DONE ||
        err === MPG123_NEW_FORMAT || err === MPG123_NEED_MORE) ||
        throw(ArgumentError("mpg123 failed while decoding: $(_mpg123_strerror(err))"))
    return Int(done[]) ÷ sizeof(Int16), err
end

# decode a whole MP3 as a mono Vector{T}, 16-bit samples scaled by 1/32768.
# Multi-channel files are averaged to mono (see `to_mono`): the whole Audio911
# analysing pipeline is intended to work exclusively on mono signals.
function _read_mp3(::Type{T}, path::String) where {T<:AbstractFloat}
    mh = _mpg123_new()
    try
        _mpg123_open(mh, path)
        rate, nch, enc = _mpg123_getformat(mh)
        enc === MPG123_ENC_SIGNED_16 || throw(ArgumentError("unsupported " *
            "mpg123 encoding $enc in '$path' (only signed 16-bit is supported)"))
        est = max(_mpg123_length(mh), 0)
        block = max(_mpg123_outblock(mh) ÷ sizeof(Int16), nch * 1152)
        buf = Vector{Int16}(undef, block)
        acc = Vector{Int16}(undef, 0)
        sizehint!(acc, est * nch)
        while true
            n, err = _mpg123_read!(mh, buf)
            n > 0 && append!(acc, view(buf, 1:n))
            err === MPG123_DONE && break
            (n === 0 && err != MPG123_NEW_FORMAT) && break
        end
        # de-interleave into (frames × channels), scaled to [-1, 1)
        nfr  = length(acc) ÷ nch
        data = Matrix{T}(undef, nfr, nch)
        scale = T(1 / 32768)
        @inbounds for c in 1:nch, i in 1:nfr
            data[i, c] = T(acc[(i - 1) * nch + c]) * scale
        end
        # mono mixdown: average channels, then continue with a Vector
        size(data, 2) > 1 && (data = to_mono(data))
        return vec(data), rate
    finally
        _mpg123_delete(mh)
    end
end
