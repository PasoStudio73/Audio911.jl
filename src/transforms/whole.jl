# ---------------------------------------------------------------------------- #
#                 whole-signal transforms on the Frames grid                   #
# ---------------------------------------------------------------------------- #
# The transforms in src/transforms/ (PWT, S-transform, fast S-transform, NSGT)
# analyse the whole signal at once and return one complex series per band.
# To plug into the pipeline they are pooled onto the time grid of a `Frames`
# object exactly like `Cwt`: the power (or magnitude) of every band is
# averaged over each frame, weighted by the frame window, so the result has
# one column per frame and can replace an `Stft` anywhere.

# pool the |y|² (or |y|) of one band series `y` (length ≥ signal length) onto
# row `row` of `spec`; `y` is indexed like `get_signal(frames)`
function _pool_band!(spec::AbstractMatrix{T}, row::Int, y::AbstractVector{<:Number},
                     frames::Frames{T}, spectrum::Base.Callable) where T
    w  = get_window(frames)
    ws = length(w)
    wsum = sum(w)
    @inbounds for (j, st) in enumerate(frames.starts)
        acc = zero(T)
        base = st - 1
        @simd for i in 1:ws
            acc += w[i] * T(spectrum(y[base + i]))
        end
        spec[row, j] = acc / wsum
    end
    return spec
end

# the value of a band series held at every frame centre (complex accessor of
# the pooled front ends)
function _centre_samples(y::AbstractVector{<:Number}, frames::Frames)
    ws = get_winsize(frames)
    return [y[st + ws ÷ 2] for st in frames.starts]
end

# centre sample (1-based, in `get_signal(frames)` coordinates) of every frame
_frame_centres(frames::Frames) = [st + get_winsize(frames) ÷ 2 for st in frames.starts]

# common accessors of the pooled front ends: they store `spec`, `freq`,
# `frames` and an `info` record with `sr`, `winsize`, `winstep`, `offset`,
# `spectrum`
macro pooled_frontend(name)
    quote
        Base.eltype(::$(esc(name)){T}) where T = T
        Audio911.get_data(s::$(esc(name)))     = s.spec
        Audio911.get_spec(s::$(esc(name)))     = s.spec
        Audio911.get_freq(s::$(esc(name)))     = s.freq
        Audio911.get_setup(s::$(esc(name)))    = s.info
        Audio911.get_sr(s::$(esc(name)))       = s.info.sr
        Audio911.get_spectrum(s::$(esc(name))) = s.info.spectrum
        Audio911.get_winsize(s::$(esc(name)))  = s.info.winsize
        Audio911.get_step(s::$(esc(name)))     = s.info.winstep
        Audio911.get_overlap(s::$(esc(name)))  = s.info.winsize - s.info.winstep
        Audio911.get_offset(s::$(esc(name)))   = s.info.offset
        Audio911.get_frames(s::$(esc(name)))   = s.frames
        Audio911.get_parent(s::$(esc(name)))   = s.frames
        Audio911.get_energy(s::$(esc(name)))   = get_energy(s.frames)
    end
end

_check_spectrum(spectrum) = spectrum in (power, magnitude) ||
    throw(ArgumentError("spectrum must be `power` or `magnitude`, got $spectrum"))
