# ---------------------------------------------------------------------------- #
#                        non-stationary Gabor transform                        #
# ---------------------------------------------------------------------------- #
# Port of audioFlux's NSGT (src/nsgt_algorithm.c, src/filterbank/
# nsgt_filterBank.c, MIT licence, Copyright (c) 2023 libAudioFlux): the FFT of
# the whole signal is cut into one window per band (window length from the
# neighbouring band edges), every windowed slice is demodulated to baseband
# and inverse-transformed at its own length, so each band has its own time
# resolution (a "cell"). `nsgt_matrix` is audioFlux's resampling of the
# cells to a common length; `Nsgt` pools them on the Frames grid.

const NSGT_STYLES = (triangular, etsi, rect, hanning, hamming, blackman, bohman, kaiser, gauss)

# window of one NSGT band: symmetric (efficient bank) or periodic (standard)
function _nsgt_window(::Type{T}, style::Base.Callable, len::Int, periodic::Bool) where T
    f = style === triangular ? triang : style === etsi ? bartlett : style
    len == 1 && return T[1]
    w = periodic ? f(len + 1)[1:len] : f(len)
    return Vector{T}(w)
end

"""
    nsgt(x, sr; nbands=84, scale=octave, style=hanning, norm=bandwidth,
         freqrange=(33, sr÷2), bins_per_octave=12, min_len=3, standard=false,
         T=eltype(x)) -> (cells, freq, lengths)

Non-stationary Gabor transform of a whole signal (audioFlux `NSGT`). The band
edges come from `scale` (see [`auditory_fbank`](@ref)) rounded to the FFT
bins of `length(x)`; band `k` takes a window of `style` shape and length
`2·max(centre − left, right − centre) + 1` (`standard=true`: `right − left + 1`,
periodic window), at least `min_len`, centred on its bin, and
`norm=bandwidth` divides the window by `sqrt(length)`. The windowed slice of
the spectrum is shifted to baseband and inverse transformed at its own
length: `cells[k]` is a complex vector of `lengths[k]` samples spanning the
whole signal duration, `freq[k]` the band centre in Hz.
"""
function nsgt(x::AbstractVector{<:Real}, sr::Int; nbands::Int=84, scale::Function=octave,
              style::Function=hanning, norm::Function=bandwidth,
              freqrange::FreqRange=(scale in (octave, logspace) ? 33 : 0, sr ÷ 2),
              bins_per_octave::Int=12, min_len::Int=3, standard::Bool=false, T::Type=float(eltype(x)))
    N = length(x)
    N ≥ 4 || throw(ArgumentError("the signal needs at least four samples"))
    scale in AVAIL_SCALES || throw(ArgumentError("scale must be one of $(AVAIL_SCALES), got $scale"))
    style in NSGT_STYLES || throw(ArgumentError("style must be one of $(NSGT_STYLES), got $style"))
    norm in (none_norm, bandwidth) || throw(ArgumentError("norm must be `none_norm` or `bandwidth`"))
    min_len ≥ 1 || throw(ArgumentError("min_len must be ≥ 1"))
    0 ≤ get_low(freqrange) < get_hi(freqrange) ≤ sr ÷ 2 || throw(ArgumentError(
        "freqrange must satisfy 0 ≤ low < high ≤ sr/2, got $freqrange"))
    edges = _band_edges(T, scale, freqrange, nbands, bins_per_octave)
    edges[end - 1] ≤ sr / 2 * (1 + 4 * eps(T)) || throw(ArgumentError(
        "the last band centre ($(round(edges[end - 1], digits=1)) Hz) is above the Nyquist frequency; check freqrange"))
    bins = [round(Int, N * Float64(e) / sr) for e in edges]
    lens = Vector{Int}(undef, nbands)
    offs = Vector{Int}(undef, nbands)
    for k in 1:nbands
        left, cur, right = bins[k], bins[k + 1], bins[k + 2]
        if standard
            len = right - left + 1
        else
            len = right - left ≥ 1 ? 2 * max(cur - left, right - cur) + 1 : 0
        end
        lens[k] = max(len, min_len)
        offs[k] = max(cur - lens[k] ÷ 2, 0)
    end
    X = fft(Vector{Complex{T}}(x))
    cells = Vector{Vector{Complex{T}}}(undef, nbands)
    for k in 1:nbands
        len = lens[k]
        w = _nsgt_window(T, style, len, standard)
        norm === bandwidth && (w ./= T(sqrt(len)))
        buf = zeros(Complex{T}, len)
        @inbounds for j in 0:len-1
            i = clamp(offs[k] + j, 0, N - 1)                # spectrum bin (0-based), clamped
            pos = mod(j + len - len ÷ 2, len)               # centre bin goes to DC
            buf[pos + 1] = X[i + 1] * w[j + 1]
        end
        cells[k] = ifft(buf)
    end
    return cells, Vector{T}(edges[2:end-1]), lens
end

# audioFlux's time axis of a cell: `len + 1` edges from -off to T + off
@inline function _nsgt_hold_index(t::Real, dur::Real, len::Int)
    det = max(len - 2, 0)
    off = dur / (len + det)
    step = (dur + 2off) / len
    # first k in 0:len with t < -off + k*step, then cell k-1
    k = floor(Int, (t + off) / step) + 1
    return clamp(k, 1, len)
end

"""
    nsgt_matrix(cells, lengths, dur, ncols) -> Matrix{Complex}

audioFlux's matrix form of an NSGT: every cell is resampled to `ncols`
columns by holding, at time `t_j = j · dur / ncols`, the last cell sample
whose (audioFlux) time edge is below `t_j`. `dur` is the signal duration in
seconds and `ncols` is normally `maximum(lengths)`.
"""
function nsgt_matrix(cells::AbstractVector{<:AbstractVector{C}}, lengths::AbstractVector{Int},
                     dur::Real, ncols::Int) where {C<:Complex}
    out = Matrix{C}(undef, length(cells), ncols)
    for (k, cell) in enumerate(cells)
        for j in 0:ncols-1
            out[k, j + 1] = cell[_nsgt_hold_index(j * dur / ncols, dur, lengths[k])]
        end
    end
    return out
end

# ---------------------------------------------------------------------------- #
#                                   front end                                  #
# ---------------------------------------------------------------------------- #
struct NsgtSetup{T<:AudioData} <: AbstractSetup
    sr              :: Int64
    winsize         :: Int64
    winstep         :: Int64
    offset          :: Int64
    spectrum        :: Base.Callable
    nbands          :: Int64
    scale           :: Base.Callable
    style           :: Base.Callable
    norm            :: Base.Callable
    freqrange       :: FreqRange
    bins_per_octave :: Int64
    min_len         :: Int64
    standard        :: Bool
end

"""
    Nsgt{T} <: AbstractSpectrogram

Non-stationary Gabor transform pooled on the time grid of a [`Frames`](@ref)
object: `nbands × frames`, power or magnitude, on the band centres of the
chosen scale. The cells (one complex series per band at the band's own time
resolution) are kept and returned by [`get_cells`](@ref). Implements the
front-end interface. See [`Nsgt(frames; kwargs...)`](@ref Nsgt(::Frames)).
"""
struct Nsgt{T<:AudioData} <: AbstractSpectrogram
    spec    :: Matrix{T}
    freq    :: Vector{T}
    frames  :: Frames{T}
    info    :: NsgtSetup{T}
    cells   :: Vector{Vector{Complex{T}}}
    lengths :: Vector{Int}
end
@pooled_frontend Nsgt
get_nbands(n::Nsgt) = n.info.nbands

"""
    get_cells(n::Nsgt) -> Vector{Vector{Complex}}

The NSGT cells: band `k` is a complex series of `get_lengths(n)[k]` samples
spanning the whole signal.
"""
get_cells(n::Nsgt) = n.cells

"""
    get_lengths(n::Nsgt) -> Vector{Int}

Number of time samples of every NSGT band.
"""
get_lengths(n::Nsgt) = n.lengths

Base.show(io::IO, n::Nsgt{T}) where T =
    print(io, "Nsgt{$T}($(size(n.spec, 2)) frames × $(size(n.spec, 1)) bands, $(nameof(n.info.scale)), cells $(minimum(n.lengths))-$(maximum(n.lengths)), sr=$(n.info.sr) Hz)")

"""
    Nsgt(frames::Frames; nbands=84, scale=octave, style=hanning, norm=bandwidth,
         freqrange, bins_per_octave=12, min_len=3, standard=false, spectrum=power) -> Nsgt

Compute [`nsgt`](@ref) on the signal of `frames` and pool every band on
the frames: the cell is expanded to the signal length by audioFlux's hold
rule and its power (or magnitude) is averaged over each frame, weighted by
the frame window.

```julia
n = Nsgt(Frames(audio; winsize=512, winstep=256); scale=octave, nbands=84)
get_cells(n)[84]           # the top band at its own resolution
```
"""
function Nsgt(frames::Frames{T}; nbands::Int=84, scale::Function=octave, style::Function=hanning,
              norm::Function=bandwidth, freqrange::FreqRange=(scale in (octave, logspace) ? 33 : 0, get_sr(frames) ÷ 2),
              bins_per_octave::Int=12, min_len::Int=3, standard::Bool=false, spectrum::Base.Callable=power) where T
    _check_spectrum(spectrum)
    sr = get_sr(frames)
    x  = get_signal(frames)
    N  = length(x)
    cells, freq, lens = nsgt(x, sr; nbands, scale, style, norm, freqrange, bins_per_octave, min_len, standard, T)
    dur = N / sr
    spec = Matrix{T}(undef, nbands, length(frames))
    y = Vector{T}(undef, N)
    for k in 1:nbands
        cell = cells[k]; len = lens[k]
        @inbounds for n in 1:N
            y[n] = T(spectrum(cell[_nsgt_hold_index((n - 1) / sr, dur, len)]))
        end
        _pool_band!(spec, k, y, frames, identity)
    end
    info = NsgtSetup{T}(sr, get_winsize(frames), get_step(frames), get_offset(frames), spectrum,
                        nbands, scale, style, norm, freqrange, bins_per_octave, min_len, standard)
    return Nsgt{T}(spec, freq, frames, info, cells, lens)
end

Nsgt(audio::AbstractVecOrMat{<:Real}, sr::Int; winsize::Int=sr ≤ 8000 ? 256 : 512, winstep::Int=winsize ÷ 2,
     type::Base.Callable=rect, periodic::Bool=true, center::Bool=false, pad_mode::Symbol=:constant, kwargs...) =
    Nsgt(Frames(audio, sr; winsize, winstep, type, periodic, center, pad_mode); kwargs...)
Nsgt(a::AudioFile; kwargs...) = Nsgt(get_data(a), get_sr(a); kwargs...)

"""
    get_complex(n::Nsgt) -> Matrix{Complex}

The cell values held at every frame centre, `bands × frames`.
"""
function get_complex(n::Nsgt{T}) where T
    N = length(get_signal(n.frames))
    dur = N / n.info.sr
    c = _frame_centres(n.frames)
    out = Matrix{Complex{T}}(undef, n.info.nbands, length(c))
    for k in 1:n.info.nbands, (j, s) in enumerate(c)
        out[k, j] = n.cells[k][_nsgt_hold_index((s - 1) / n.info.sr, dur, n.lengths[k])]
    end
    return out
end
