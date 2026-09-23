# ---------------------------------------------------------------------------- #
#                                   windows                                    #
# ---------------------------------------------------------------------------- #
"""
    povey(n::Integer) -> Vector{Float64}

Kaldi's default analysis window, a Hann window raised to the power 0.85:
`(0.5 - 0.5cos(2πk/(n-1)))^0.85`. It is symmetric, like Kaldi builds it.
"""
povey(n::Integer) = [(0.5 - 0.5 * cospi(2k / (n - 1)))^0.85 for k in 0:n-1]

# availables window functions from package DSP, plus povey
const AVAIL_WINDOWS = (
    rect, hanning, hamming, cosine, lanczos, triang,
    bartlett, bartlett_hann, blackman, povey
)

# a periodic window of length n is the symmetric window of length n+1 without
# its last sample; this is MATLAB's `"periodic"` definition for every n.
function _make_window(::Type{T}, type::Base.Callable, n::Int, periodic::Bool) where {T<:AudioData}
    in(type, AVAIL_WINDOWS) || throw(ArgumentError(
        "Window type $(type) not supported. Available windows: $(AVAIL_WINDOWS)"))
    w = periodic ? type(n + 1)[1:n] : type(n)
    return Vector{T}(w)
end

"""
    movingwindow(; winsize, winstep=winsize ÷ 2) -> NamedTuple

Framing parameters as a named tuple, accepted by the `win` keyword of
[`Frames`](@ref), [`Stft`](@ref) and [`Cwt`](@ref):
`Stft(audio; win=movingwindow(winsize=512, winstep=256))` is the same as
`Stft(audio; winsize=512, winstep=256)`. Kept for compatibility with the
DataTreatments-based API.
"""
movingwindow(; winsize::Int64, winstep::Int64=winsize ÷ 2) = (; winsize, winstep)

# resolve the `win` keyword against explicit winsize/winstep
function _winparams(win, winsize, winstep)
    isnothing(win) && return winsize, winstep
    ws = get(win, :winsize, winsize)
    st = get(win, :winstep, ws ÷ 2)
    return ws, st
end

# ---------------------------------------------------------------------------- #
#                                 pre-emphasis                                 #
# ---------------------------------------------------------------------------- #
"""
    preemphasis(x::AbstractVector; coef=0.97, zi=x[1]) -> Vector

First-order high-pass pre-emphasis `y[n] = x[n] - coef * x[n-1]`.

`zi` is the virtual sample preceding `x[1]`; the default `x[1]` gives
`y[1] = (1 - coef) * x[1]`, the HTK and Kaldi convention. Use `zi=0` for the
python_speech_features convention (`y[1] = x[1]`) and
`zi=-(2x[1] - x[2]) / coef` to reproduce librosa's default initial state.

Per-frame pre-emphasis (HTK, Kaldi, ETSI) is applied by [`Frames`](@ref)
with its `preemph` keyword; this function is the signal-level (librosa) form.

See also [`deemphasis`](@ref).
"""
function preemphasis(x::AbstractVector{T}; coef::Real=0.97, zi::Real=x[1]) where {T<:Real}
    y = similar(x, float(T))
    k = float(T)(coef)
    prev = float(T)(zi)
    @inbounds for i in eachindex(x)
        y[i] = x[i] - k * prev
        prev = x[i]
    end
    return y
end

"""
    deemphasis(y::AbstractVector; coef=0.97, zi=0) -> Vector

Inverse of [`preemphasis`](@ref): `x[n] = y[n] + coef * x[n-1]`, with `zi` the
value of `x[0]`.
"""
function deemphasis(y::AbstractVector{T}; coef::Real=0.97, zi::Real=0) where {T<:Real}
    x = similar(y, float(T))
    k = float(T)(coef)
    prev = float(T)(zi)
    @inbounds for i in eachindex(y)
        prev = y[i] + k * prev
        x[i] = prev
    end
    return x
end

# ---------------------------------------------------------------------------- #
#                                     info                                     #
# ---------------------------------------------------------------------------- #
struct FramesSetup <: AbstractSetup
    sr         :: Int64
    winsize    :: Int64
    winstep    :: Int64
    type       :: Base.Callable
    periodic   :: Bool
    center     :: Bool
    pad_mode   :: Symbol
    preemph    :: Float64
    dc_removal :: Bool
    pad_end    :: Bool
    offset     :: Int64
end

# ---------------------------------------------------------------------------- #
#                                    Frames                                    #
# ---------------------------------------------------------------------------- #
"""
    Frames{T} <: AbstractFrame

A framed mono signal. The frames are **lazy**: the object stores the signal,
the start index of every frame and the analysis window; consumers stream
through the frames with a single buffer ([`frame!`](@ref)) and
[`get_data`](@ref) materialises the `winsize × nframes` matrix on demand.

Build one with [`Frames(audio; kwargs...)`](@ref Frames(::AudioFile)).
"""
struct Frames{T<:AudioData} <: AbstractFrame
    signal :: Vector{T}
    starts :: StepRange{Int64,Int64}
    window :: Vector{T}
    info   :: FramesSetup
end

# ---------------------------------------------------------------------------- #
#                                    methods                                   #
# ---------------------------------------------------------------------------- #
Base.length(f::Frames) = length(f.starts)
Base.eltype(::Frames{T}) where T = T

"""
    get_size(f::Frames) -> Int

Frame length in samples.
"""
get_size(f::Frames)    = f.info.winsize
get_winsize(f::Frames) = f.info.winsize

"""
    get_step(f::Frames) -> Int

Hop between consecutive frames in samples.
"""
get_step(f::Frames)    = f.info.winstep

"""
    get_overlap(f::Frames) -> Int

Overlap between consecutive frames in samples (`winsize - winstep`).
"""
get_overlap(f::Frames) = get_size(f) - get_step(f)
get_sr(f::Frames)      = f.info.sr
get_setup(f::Frames)   = f.info
get_offset(f::Frames)  = f.info.offset
get_nframes(f::Frames) = length(f)

"""
    get_window(f::Frames) -> Vector

The analysis window, already converted to the element type of the frames.
"""
get_window(f::Frames)  = f.window

"""
    get_signal(f::Frames) -> Vector

The mono signal the frames were cut from (padded when `center=true`).
"""
get_signal(f::Frames)  = f.signal

"""
    frame!(buf, f::Frames, i) -> buf

Write frame `i` (raw, after optional DC removal and pre-emphasis, before
windowing) into `buf`, which must have length `get_size(f)`.
"""
@inline function frame!(buf::AbstractVector{T}, f::Frames{T}, i::Int) where T
    n = f.info.winsize
    s = f.starts[i]
    x = f.signal
    @inbounds copyto!(buf, 1, x, s, n)
    if f.info.dc_removal
        μ = zero(T)
        @inbounds @simd for j in 1:n; μ += buf[j]; end
        μ /= n
        @inbounds @simd for j in 1:n; buf[j] -= μ; end
    end
    k = T(f.info.preemph)
    if !iszero(k)
        @inbounds for j in n:-1:2; buf[j] -= k * buf[j-1]; end
        @inbounds buf[1] *= (one(T) - k)
    end
    return buf
end

"""
    get_data(f::Frames) -> Matrix

Materialise the frames as a `winsize × nframes` matrix (one frame per column),
with DC removal and pre-emphasis applied but without the window.
"""
function get_data(f::Frames{T}) where T
    n = length(f)
    out = Matrix{T}(undef, f.info.winsize, n)
    for i in 1:n
        frame!(view(out, :, i), f, i)
    end
    return out
end

"""
    get_winframes(f::Frames) -> Matrix

Materialise the windowed frames (`get_data(f) .* get_window(f)`).
"""
function get_winframes(f::Frames{T}) where T
    out = get_data(f)
    out .*= f.window
    return out
end

"""
    get_energy(f::Frames) -> Vector

Raw energy `sum(x.^2)` of every frame, computed after DC removal and before
pre-emphasis and windowing (HTK `RAWENERGY`, Kaldi `raw_energy`).
"""
function get_energy(f::Frames{T}) where T
    n = f.info.winsize
    x = f.signal
    e = Vector{T}(undef, length(f))
    @inbounds for (i, s) in enumerate(f.starts)
        acc = zero(T)
        if f.info.dc_removal
            μ = zero(T)
            @simd for j in s:s+n-1; μ += x[j]; end
            μ /= n
            @simd for j in s:s+n-1; acc += (x[j] - μ)^2; end
        else
            @simd for j in s:s+n-1; acc += x[j]^2; end
        end
        e[i] = acc
    end
    return e
end

# ---------------------------------------------------------------------------- #
#                                   base.show                                  #
# ---------------------------------------------------------------------------- #
function Base.show(io::IO, ::MIME"text/plain", f::Frames{T}) where T
    n_frames   = length(f)
    frame_size = get_size(f)
    step_size  = get_step(f)
    overlap    = get_overlap(f)

    println(io, "Frames{$T}")
    println(io, "  Sample rate: $(f.info.sr) Hz")
    println(io, "  Frames:      $n_frames")
    println(io, "  Frame size:  $frame_size samples")
    println(io, "  Step:        $step_size samples")
    println(io, "  Overlap:     $overlap samples ($(round(100 * overlap / frame_size, digits=1))%)")
    println(io, "  Window:      $(f.info.type)$(f.info.periodic ? " (periodic)" : "")")
    f.info.center && println(io, "  Centered:    $(f.info.pad_mode) padding")
    iszero(f.info.preemph) || println(io, "  Pre-emphasis: $(f.info.preemph)")
    f.info.dc_removal && println(io, "  DC removal:  true")
end

function Base.show(io::IO, f::Frames{T}) where T
    print(io, "Frames{$T}($(length(f)) frames × $(get_size(f)) samples)")
end

# ---------------------------------------------------------------------------- #
#                                    helpers                                   #
# ---------------------------------------------------------------------------- #
# mono, concrete vector of type T (no copy when already so)
_to_mono(x::Vector{T}) where {T<:AudioData} = x
_to_mono(x::AbstractVector{T}) where {T<:AudioData} = Vector{T}(x)
function _to_mono(x::AbstractMatrix{T}) where {T<:AudioData}
    size(x, 2) == 1 && return _to_mono(vec(x))
    return vec(mean(x, dims=2))
end
_to_mono(x::AbstractVecOrMat{<:Real}) = _to_mono(Float32.(x))

function _pad_center(x::Vector{T}, pad::Int, mode::Symbol) where T
    n = length(x)
    y = Vector{T}(undef, n + 2pad)
    copyto!(y, pad + 1, x, 1, n)
    if mode == :constant
        fill!(view(y, 1:pad), zero(T))
        fill!(view(y, n+pad+1:n+2pad), zero(T))
    elseif mode == :reflect
        pad < n || throw(ArgumentError("reflect padding needs a signal longer than $pad samples"))
        @inbounds for i in 1:pad
            y[pad + 1 - i] = x[i + 1]
            y[n + pad + i] = x[n - i]
        end
    elseif mode == :edge
        fill!(view(y, 1:pad), x[1])
        fill!(view(y, n+pad+1:n+2pad), x[n])
    else
        throw(ArgumentError("pad_mode must be :constant, :reflect or :edge, got $mode"))
    end
    return y
end

# ---------------------------------------------------------------------------- #
#                                    frames                                    #
# ---------------------------------------------------------------------------- #
"""
    Frames(audio::AudioFile; kwargs...) -> Frames
    Frames(x::AbstractVecOrMat, sr::Int; kwargs...) -> Frames

Cut a mono signal into (overlapping) frames and attach an analysis window.
A multi-channel matrix is averaged to mono first.

# Keyword Arguments
- `winsize::Int`: frame length in samples (default 256 for `sr ≤ 8000`, else 512)
- `winstep::Int`: hop in samples (default `winsize ÷ 2`)
- `win`: alternatively, both as `movingwindow(winsize=..., winstep=...)`
- `type::Base.Callable=hanning`: window function, one of `rect`, `hanning`,
  `hamming`, `cosine`, `lanczos`, `triang`, `bartlett`, `bartlett_hann`,
  `blackman`, `povey`
- `periodic::Bool=true`: periodic (DFT-even) window, MATLAB's `"periodic"`;
  `false` gives the symmetric window
- `center::Bool=false`: pad the signal by `winsize ÷ 2` on both sides so that
  frame `i` is centred on sample `(i-1) * winstep` (librosa's `center=True`)
- `pad_mode::Symbol=:constant`: padding used when `center=true`
  (`:constant`, `:reflect` or `:edge`)
- `preemph::Real=0`: per-frame pre-emphasis coefficient, HTK/Kaldi style
  (`0.97` is the usual value; `0` disables it)
- `dc_removal::Bool=false`: subtract the mean of every frame (Kaldi's
  `remove_dc_offset`)
- `pad_end::Bool=false`: zero-pad the end of the signal so that a trailing
  segment shorter than `winsize` still yields a frame (python_speech_features)

Frames are taken from the start of the signal with the given hop; a trailing
segment shorter than `winsize` is dropped (MATLAB behaviour) unless
`pad_end=true`.

# Examples
```julia
audio  = load("speech.wav"; sr=16000)
frames = Frames(audio; winsize=512, winstep=256, type=hamming, periodic=true)
length(frames)        # number of frames
get_data(frames)      # 512 × nframes matrix
get_window(frames)    # the Hamming window as a Vector{Float32}

# Kaldi-style framing
frames = Frames(audio; winsize=400, winstep=160, type=povey, periodic=false,
                preemph=0.97, dc_removal=true)
```
"""
function Frames(
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
)
    winsize, winstep = _winparams(win, winsize, winstep)
    x = _to_mono(audio)
    T = eltype(x)

    winsize > 0 || throw(ArgumentError("winsize must be positive, got $winsize"))
    winstep > 0 || throw(ArgumentError("winstep must be positive, got $winstep"))
    winstep ≤ winsize || throw(ArgumentError(
        "winstep ($winstep) must not exceed winsize ($winsize)"))
    0 ≤ preemph < 1 || throw(ArgumentError("preemph must be in [0, 1), got $preemph"))

    offset = 0
    if center
        pad = winsize ÷ 2
        x = _pad_center(x, pad, pad_mode)
        offset = -pad
    end

    n = length(x)
    if pad_end
        # enough zeros for the last partial frame to become a full one
        nfr = n ≤ winsize ? 1 : 1 + cld(n - winsize, winstep)
        need = (nfr - 1) * winstep + winsize
        need > n && (x = vcat(x, zeros(T, need - n)); n = need)
    end
    n ≥ winsize || throw(ArgumentError(
        "Audio length ($n samples) is shorter than window size ($winsize)"))

    starts = 1:winstep:(n - winsize + 1)
    window = _make_window(T, type, winsize, periodic)
    info   = FramesSetup(sr, winsize, winstep, type, periodic, center, pad_mode,
                         Float64(preemph), dc_removal, pad_end, offset)

    return Frames{T}(x, starts, window, info)
end

Frames(a::AudioFile; kwargs...) = Frames(get_data(a), get_sr(a); kwargs...)
