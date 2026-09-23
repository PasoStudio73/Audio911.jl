# ---------------------------------------------------------------------------- #
#                            pseudo wavelet transform                          #
# ---------------------------------------------------------------------------- #
# Port of audioFlux's PWT (src/pwt_algorithm.c, MIT licence, Copyright (c)
# 2023 libAudioFlux): the FFT of the whole signal is multiplied by a bank of
# bandpass filters designed on any frequency scale with any filter style, and
# every product is inverse-transformed, giving one complex analytic series
# per band.

"""
    pwt(x, sr; nbands=84, scale=octave, style=triangular, norm=none_norm,
        freqrange, bins_per_octave=12, T=eltype(x)) -> (Y, freq)

Pseudo wavelet transform of a whole signal (audioFlux `PWT`): `Y` is the
`nbands × length(x)` complex matrix whose row `k` is the inverse FFT of the
signal spectrum multiplied by band `k` of an [`auditory_fbank`](@ref)
designed on the `length(x)`-point FFT grid; `freq` holds the band centres.
Only the positive-frequency half of the spectrum is filtered, so every row
is an analytic signal. Keywords are those of `auditory_fbank`; the default
scale is `octave` from C1 with the range `(32.703, sr/2)`.
"""
function pwt(x::AbstractVector{<:Real}, sr::Int; nbands::Int=84, scale::Function=octave,
             style::Function=triangular, norm::Function=none_norm,
             freqrange::FreqRange=(scale in (octave, logspace) ? 33 : 0, sr ÷ 2),
             bins_per_octave::Int=12, T::Type=float(eltype(x)))
    N  = length(x)
    N ≥ 2 || throw(ArgumentError("the signal needs at least two samples"))
    half = N ÷ 2 + 1
    sfreq = T.((0:half-1) .* (sr / N))
    fb = auditory_fbank(sr; sfreq, nbands, scale, style, norm, freqrange, bins_per_octave)
    W  = get_data(fb)
    X  = fft(Vector{Complex{T}}(x))
    Y  = Matrix{Complex{T}}(undef, nbands, N)
    plan = plan_ifft!(zeros(Complex{T}, N))
    Threads.@threads for chunk in _chunks(nbands)
        buf = zeros(Complex{T}, N)
        for k in chunk
            fill!(buf, zero(Complex{T}))
            @inbounds for i in 1:half
                buf[i] = X[i] * W[k, i]
            end
            plan * buf
            @inbounds Y[k, :] .= buf
        end
    end
    return Y, get_freq(fb)
end

struct PwtSetup{T<:AudioData} <: AbstractSetup
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
end

"""
    Pwt{T} <: AbstractSpectrogram

Pseudo wavelet transform of a signal pooled on the time grid of a
[`Frames`](@ref) object (audioFlux `PWT`): `nbands × frames`, power or
magnitude, on the frequency grid of the filterbank. Implements the front-end
interface. See [`Pwt(frames; kwargs...)`](@ref Pwt(::Frames)).
"""
struct Pwt{T<:AudioData} <: AbstractSpectrogram
    spec   :: Matrix{T}
    freq   :: Vector{T}
    frames :: Frames{T}
    info   :: PwtSetup{T}
end
@pooled_frontend Pwt

get_nbands(p::Pwt) = p.info.nbands
Base.show(io::IO, p::Pwt{T}) where T =
    print(io, "Pwt{$T}($(size(p.spec, 2)) frames × $(size(p.spec, 1)) bands, $(nameof(p.info.scale)), sr=$(p.info.sr) Hz)")

"""
    Pwt(frames::Frames; nbands=84, scale=octave, style=triangular, norm=none_norm,
        freqrange, bins_per_octave=12, spectrum=power) -> Pwt

Compute [`pwt`](@ref) on the signal of `frames` and pool the power (or
magnitude) of every band over the frames, weighted by the frame window,
like [`Cwt`](@ref). Keywords are those of [`auditory_fbank`](@ref).

```julia
frames = Frames(audio; winsize=512, winstep=256, type=rect)
p = Pwt(frames; scale=octave, nbands=84, style=hanning)
mel = MelSpec(p; nbands=20, freqrange=(50, 3900))
```
"""
function Pwt(frames::Frames{T}; nbands::Int=84, scale::Function=octave, style::Function=triangular,
             norm::Function=none_norm, freqrange::FreqRange=(scale in (octave, logspace) ? 33 : 0, get_sr(frames) ÷ 2),
             bins_per_octave::Int=12, spectrum::Base.Callable=power) where T
    _check_spectrum(spectrum)
    sr = get_sr(frames)
    Y, freq = pwt(get_signal(frames), sr; nbands, scale, style, norm, freqrange, bins_per_octave, T)
    spec = Matrix{T}(undef, nbands, length(frames))
    for k in 1:nbands
        _pool_band!(spec, k, view(Y, k, :), frames, spectrum)
    end
    info = PwtSetup{T}(sr, get_winsize(frames), get_step(frames), get_offset(frames), spectrum,
                       nbands, scale, style, norm, freqrange, bins_per_octave)
    return Pwt{T}(spec, Vector{T}(freq), frames, info)
end

Pwt(audio::AbstractVecOrMat{<:Real}, sr::Int; winsize::Int=sr ≤ 8000 ? 256 : 512, winstep::Int=winsize ÷ 2,
    type::Base.Callable=rect, periodic::Bool=true, center::Bool=false, pad_mode::Symbol=:constant, kwargs...) =
    Pwt(Frames(audio, sr; winsize, winstep, type, periodic, center, pad_mode); kwargs...)
Pwt(a::AudioFile; kwargs...) = Pwt(get_data(a), get_sr(a); kwargs...)

"""
    get_complex(p::Pwt) -> Matrix{Complex}

The complex band series sampled at the frame centres, `bands × frames`
(recomputed with [`pwt`](@ref)).
"""
function get_complex(p::Pwt{T}) where T
    i = p.info
    Y, _ = pwt(get_signal(p.frames), i.sr; nbands=i.nbands, scale=i.scale, style=i.style, norm=i.norm,
               freqrange=i.freqrange, bins_per_octave=i.bins_per_octave, T)
    c = _frame_centres(p.frames)
    return Y[:, c]
end
