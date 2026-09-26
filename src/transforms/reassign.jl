# ---------------------------------------------------------------------------- #
#                         reassigned STFT spectrogram                          #
# ---------------------------------------------------------------------------- #
# Port of audioFlux's reassign (src/reassign_algorithm.c, MIT licence,
# Copyright (c) 2023 libAudioFlux), after Auger & Flandrin (1995): every STFT
# cell is moved to its local centre of gravity, the instantaneous frequency
# f - Im(S_dh / S_h)·sr/2π and the group delay t + Re(S_th / S_h)/sr, where
# S_dh and S_th are the STFTs with the window derivative and the
# time-weighted window.

struct ReassignSetup <: AbstractSetup
    sr         :: Int64
    mode       :: Symbol
    thresh     :: Float64
    order      :: Int64
    accumulate :: Symbol
    spectrum   :: Base.Callable
end

"""
    Reassign{F,T} <: AbstractSpectrogram

Reassigned spectrogram of an [`Stft`](@ref), on the same frequency and time
grid (`bins × frames`). [`get_reassigned`](@ref) gives the reassigned
frequency and time of every cell. It implements the front-end interface.
See [`Reassign(stft; kwargs...)`](@ref Reassign(::Stft)).
"""
struct Reassign{F,T<:AbstractFloat} <: AbstractSpectrogram
    spec   :: Matrix{T}
    freqs  :: Matrix{T}
    times  :: Matrix{T}
    parent :: F
    info   :: ReassignSetup
    cplx   :: Maybe{Matrix{Complex{T}}}
end

Base.eltype(::Reassign{F,T}) where {F,T} = T
get_data(r::Reassign)     = r.spec
get_spec(r::Reassign)     = r.spec
get_freq(r::Reassign)     = get_freq(r.parent)
get_setup(r::Reassign)    = r.info
get_sr(r::Reassign)       = r.info.sr
get_spectrum(r::Reassign) = r.info.spectrum
get_parent(r::Reassign)   = r.parent
get_window(r::Reassign)   = get_window(r.parent)
get_nfft(r::Reassign)     = get_nfft(r.parent)
get_frontend(r::Reassign) = r
_freq_indices(r::Reassign, fr::FreqRange) = _freq_indices(r.parent, fr)

"""
    get_reassigned(r::Reassign) -> (freqs, times)

Reassigned frequency (Hz) and time (s, on the [`get_times`](@ref) axis) of
every cell of the original STFT, `bins × frames` each (librosa
`reassigned_spectrogram`'s `freqs` and `times`). Cells below the threshold
keep their grid position.
"""
get_reassigned(r::Reassign) = (r.freqs, r.times)

"""
    get_complex(r::Reassign) -> Matrix{Complex}

The reassigned complex values (`accumulate=:complex` only).
"""
function get_complex(r::Reassign)
    isnothing(r.cplx) && throw(ArgumentError(
        "this reassigned spectrogram accumulated energies; build it with accumulate=:complex"))
    return r.cplx
end

Base.show(io::IO, r::Reassign{F,T}) where {F,T} =
    print(io, "Reassign{$(nameof(F)),$T}($(size(r.spec, 2)) frames × $(size(r.spec, 1)) bins, mode=:$(r.info.mode), order=$(r.info.order))")

# derivative (central differences, circular ends, audioFlux) and time-weighted
# (n - N/2) windows of `w`
function _reassign_windows(w::AbstractVector{T}) where T
    N  = length(w)
    ext = vcat(w[end], w, w[1])
    dh = T[(ext[i + 2] - ext[i]) / 2 for i in 1:N]
    th = T[(n - N ÷ 2) * w[n + 1] for n in 0:N-1]
    return dh, th
end

"""
    Reassign(stft::Stft; mode=:all, thresh=0.001, order=1, accumulate=:complex,
             spectrum=get_spectrum(stft)) -> Reassign

Reassigned spectrogram (Auger & Flandrin 1995; audioFlux `reassign`). For
every cell of the complex STFT `S_h` the instantaneous frequency
`f - Im(S_dh/S_h)·sr/2π` (`S_dh`: STFT with the window derivative) and the
group delay `t + Re(S_th/S_h)/sr` (`S_th`: STFT with the window times
`n - N/2`) are computed; cells with `|S_h| < thresh` stay in place. Both are
clipped to the grid, rounded to the nearest bin and frame, and the cell is
added there.

# Keyword Arguments
- `mode::Symbol=:all`: reassign in `:all` (time and frequency), `:freq` or
  `:time` only
- `thresh::Real=0.001`: magnitude below which a cell is not moved
- `order::Int=1`: `order > 1` repeats the frequency reassignment (audioFlux
  `set_order`), sharpening further
- `accumulate::Symbol=:complex`: `:complex` adds the complex values, re-
  referenced to the window centre, then takes the power or magnitude
  (audioFlux's default); `:energy` adds the power (or magnitude) of the
  cells, which preserves the total energy (audioFlux `result_type=1` for a
  magnitude spectrum)
- `spectrum`: `power` or `magnitude` of the result

```julia
stft = Stft(audio; winsize=512, winstep=128, type=hanning)
r = Reassign(stft)
f, t = get_reassigned(r)
mel = MelSpec(r; nbands=40)          # any downstream stage
```
"""
function Reassign(s::Stft{T}; mode::Symbol=:all, thresh::Real=0.001, order::Int=1,
                  accumulate::Symbol=:complex, spectrum::Base.Callable=get_spectrum(s)) where T
    mode in (:all, :freq, :time) || throw(ArgumentError("mode must be :all, :freq or :time, got :$mode"))
    accumulate in (:complex, :energy) || throw(ArgumentError("accumulate must be :complex or :energy, got :$accumulate"))
    order ≥ 1 || throw(ArgumentError("order must be ≥ 1"))
    thresh ≥ 0 || throw(ArgumentError("thresh must be ≥ 0"))
    _check_spectrum(spectrum)
    frames = get_frames(s)
    nfft = get_nfft(s)
    sr   = get_sr(s)
    hop  = get_step(s)
    nb, nf = get_nbins(s), get_nframes(s)
    w  = get_window(frames)
    dh, th = _reassign_windows(w)
    S = get_complex(s)
    Sd = mode === :time ? nothing : _stft!(Matrix{Complex{T}}(undef, nb, nf), frames, nfft, identity, dh)
    St_ = mode === :freq ? nothing : _stft!(Matrix{Complex{T}}(undef, nb, nf), frames, nfft, identity, th)

    grid  = get_freq(s)
    times = get_times(s)
    fmax  = T(sr) / 2
    th2   = T(thresh)^2
    F  = Matrix{T}(undef, nb, nf)            # reassigned frequency, Hz
    Tm = Matrix{T}(undef, nb, nf)            # reassigned time, s
    fi = Matrix{Int}(undef, nb, nf)          # target bin
    ti = Matrix{Int}(undef, nb, nf)          # target frame
    @inbounds for j in 1:nf, k in 1:nb
        x = S[k, j]
        moved = abs2(x) ≥ th2
        f = grid[k]
        if mode !== :time && moved
            f = grid[k] - imag(Sd[k, j] / x) * T(sr) / T(2π)
        end
        dt = zero(T)
        if mode !== :freq && moved
            dt = real(St_[k, j] / x) / T(sr)
        end
        f = clamp(f, zero(T), fmax)
        F[k, j] = f
        fi[k, j] = round(Int, f * nfft / sr) + 1
        tj = clamp((j - 1) * hop / T(sr) + dt, zero(T), (nf - 1) * hop / T(sr))
        ti[k, j] = round(Int, tj * sr / hop) + 1
        Tm[k, j] = times[1] + tj
    end
    # repeated frequency reassignment within each frame
    if order > 1
        tmp = similar(fi)
        for _ in 2:order
            @inbounds for j in 1:nf, k in 1:nb
                tmp[k, j] = fi[fi[k, j], j]
            end
            fi, tmp = tmp, fi
        end
    end
    # the cells are re-referenced to the window centre before complex addition
    c = length(w) / 2
    shift = Complex{T}[cis(T(2π) * (k - 1) * T(c) / nfft) for k in 1:nb]
    if accumulate === :complex
        A = zeros(Complex{T}, nb, nf)
        @inbounds for j in 1:nf, k in 1:nb
            A[fi[k, j], ti[k, j]] += S[k, j] * shift[k]
        end
        spec = spectrum(A)
    else
        A = nothing
        spec = zeros(T, nb, nf)
        @inbounds for j in 1:nf, k in 1:nb
            spec[fi[k, j], ti[k, j]] += spectrum(S[k, j])
        end
    end
    info = ReassignSetup(sr, mode, Float64(thresh), order, accumulate, spectrum)
    return Reassign{typeof(s),T}(spec, F, Tm, s, info, A)
end
