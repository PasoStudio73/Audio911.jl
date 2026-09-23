# ---------------------------------------------------------------------------- #
#          audioFlux features: xxcc, deconvolution, cepstrogram, EZR, HR       #
# ---------------------------------------------------------------------------- #
# Ports of audioFlux's feature/xxcc, feature/deconv, cepstrogram, temporal
# and mir/harmonicRatio (MIT licence, Copyright (c) 2023 libAudioFlux).

# ---------------------------------------------------------------------------- #
#                                 xxcc_standard                                #
# ---------------------------------------------------------------------------- #
"""
    xxcc_standard(spec; ncoeffs=13, energy=spectrum_energy, energy_mode=:replace,
                  delta_length=9, rect=mlog) -> (cepstrum, delta, deltadelta)

audioFlux's `xxcc_standard` (and `mfcc_standard`): the cepstral coefficients
of any spectrogram ([`Mfcc`](@ref) with `rect=mlog`, a `1e-8` floor and the
orthonormal DCT), with the natural-log energy in place of C0
(`energy_mode=:replace`), in front of the coefficients (`:prepend`,
audioFlux's `APPEND`) or not at all (`energy=nothing`), followed by the
delta and delta-delta taken along the coefficient axis (audioFlux's
convention, [`Delta`](@ref) with `source=:transposed`). `energy` is a
function of the spectrogram ([`raw_energy`](@ref), [`spectrum_energy`](@ref))
or a vector with one value per frame.
"""
function xxcc_standard(s::AbstractSpectrogram; ncoeffs::Int=13, energy=spectrum_energy,
                       energy_mode::Symbol=:replace, delta_length::Int=9, rect::Base.Callable=mlog)
    en = energy isa AbstractVector ? (_ -> energy) : energy
    c  = Mfcc(s; ncoeffs, rect, floor=1e-8, dct=dct_ortho, energy=en, energy_mode, energy_floor=1e-8)
    d1 = Delta(c; delta_length, source=:transposed)
    d2 = Delta(d1; delta_length, source=:transposed)
    return c, d1, d2
end

# ---------------------------------------------------------------------------- #
#                                spectral deconvolution                        #
# ---------------------------------------------------------------------------- #
struct DeconvSetup <: AbstractSetup
    sr::Int64
end

"""
    Deconv{F,T} <: AbstractAudioSpectrum

Spectral deconvolution of a spectrogram into a timbre (formant) part and a
pitch (fine structure) part, audioFlux `deconv`; both `bins × frames`.
[`get_timbre`](@ref) and [`get_pitch`](@ref) return them, `get_data` the
timbre part.
"""
struct Deconv{F,T<:AudioData} <: AbstractAudioSpectrum
    timbre :: Matrix{T}
    pitch  :: Matrix{T}
    parent :: F
    info   :: DeconvSetup
end
Base.eltype(::Deconv{F,T}) where {F,T} = T
get_data(d::Deconv)   = d.timbre
get_spec(d::Deconv)   = d.timbre
get_parent(d::Deconv) = d.parent
get_sr(d::Deconv)     = d.info.sr
get_freq(d::Deconv)   = get_freq(d.parent)
get_nframes(d::Deconv) = size(d.timbre, 2)

"""
    get_timbre(d::Deconv) -> Matrix

The timbre (formant) part, `real(ifft(|F|))` of every frame.
"""
get_timbre(d::Deconv) = d.timbre

"""
    get_pitch(d::Deconv) -> Matrix

The pitch (fine structure) part, `real(ifft(F / |F|))` of every frame.
"""
get_pitch(d::Deconv)  = d.pitch
Base.show(io::IO, d::Deconv{F,T}) where {F,T} =
    print(io, "Deconv{$(nameof(F)),$T}($(size(d.timbre, 2)) frames × $(size(d.timbre, 1)) bins)")

"""
    Deconv(spec::AbstractSpectrogram) -> Deconv

Deconvolve every frame of a spectrogram (audioFlux `deconv`): the frame's
`n` bins, zero-padded to the next power of two above `2n`, are Fourier
transformed into `F`; the first `n` values of `real(ifft(|F|))` are the
timbre (a smooth, formant-like envelope) and those of
`real(ifft(F / max(|F|, 10⁻¹⁶)))` the pitch part (the fine harmonic
structure). audioFlux applies it to a magnitude spectrogram.
"""
function Deconv(s::AbstractSpectrogram)
    T = eltype(s)
    S = get_spec(s)
    n, nf = size(S)
    L = nextpow(2, 2n)
    timbre = Matrix{T}(undef, n, nf); pitch = Matrix{T}(undef, n, nf)
    buf = zeros(Complex{T}, L)
    pf = plan_fft!(buf); pi_ = plan_ifft!(copy(buf))
    tmp = similar(buf)
    for j in 1:nf
        fill!(buf, zero(Complex{T}))
        @inbounds for k in 1:n; buf[k] = S[k, j]; end
        pf * buf
        @inbounds for k in 1:L; tmp[k] = abs(buf[k]); end
        pi_ * tmp
        @inbounds for k in 1:n; timbre[k, j] = real(tmp[k]); end
        @inbounds for k in 1:L; buf[k] /= max(abs(buf[k]), T(1e-16)); end
        pi_ * buf
        @inbounds for k in 1:n; pitch[k, j] = real(buf[k]); end
    end
    return Deconv{typeof(s),T}(timbre, pitch, s, DeconvSetup(get_sr(s)))
end

# ---------------------------------------------------------------------------- #
#                                  cepstrogram                                 #
# ---------------------------------------------------------------------------- #
struct CepstrogramSetup <: AbstractSetup
    sr::Int64
    ncep::Int64
    nfft::Int64
end

"""
    Cepstrogram{T} <: AbstractAudioSpectrum

Real cepstrum of every frame and its liftered spectral envelope and
details (audioFlux `Cepstrogram`). `get_data` is the cepstrum
(`nfft ÷ 2 + 1` quefrencies × frames, [`get_quefrency`](@ref) in seconds);
[`get_envelope`](@ref) and [`get_details`](@ref) are log-power spectra on
the STFT grid ([`get_freq`](@ref)).
"""
struct Cepstrogram{T<:AudioData} <: AbstractAudioSpectrum
    cepstrum :: Matrix{T}
    envelope :: Matrix{T}
    details  :: Matrix{T}
    frames   :: Frames{T}
    info     :: CepstrogramSetup
end
Base.eltype(::Cepstrogram{T}) where T = T
get_data(c::Cepstrogram)    = c.cepstrum
get_spec(c::Cepstrogram)    = c.cepstrum
get_parent(c::Cepstrogram)  = c.frames
get_frames(c::Cepstrogram)  = c.frames
get_sr(c::Cepstrogram)      = c.info.sr
get_nframes(c::Cepstrogram) = size(c.cepstrum, 2)
get_freq(c::Cepstrogram{T}) where T = T[(k - 1) * c.info.sr / c.info.nfft for k in 1:size(c.envelope, 1)]

"""
    get_quefrency(c::Cepstrogram) -> Vector

Quefrency of every cepstral coefficient in seconds, `(k-1)/sr`.
"""
get_quefrency(c::Cepstrogram{T}) where T = T[(k - 1) / c.info.sr for k in 1:size(c.cepstrum, 1)]

"""
    get_envelope(c::Cepstrogram) -> Matrix

The spectral envelope: the log power spectrum rebuilt from the first
`ncep` cepstral coefficients (and C0), `bins × frames`.
"""
get_envelope(c::Cepstrogram) = c.envelope

"""
    get_details(c::Cepstrogram) -> Matrix

The spectral details: the log power spectrum rebuilt from the cepstral
coefficients above `ncep`, `bins × frames`.
"""
get_details(c::Cepstrogram)  = c.details
Base.show(io::IO, c::Cepstrogram{T}) where T =
    print(io, "Cepstrogram{$T}($(size(c.cepstrum, 2)) frames, nfft $(c.info.nfft), ncep $(c.info.ncep))")

"""
    Cepstrogram(frames::Frames; ncep=4) -> Cepstrogram

Cepstrogram of `frames` (audioFlux `Cepstrogram`): the windowed frame (FFT
of `winsize` samples) gives the power spectrum `P`, floored at `10⁻¹⁶`; the
real cepstrum is `real(ifft(log P))`. Keeping the quefrencies `0:ncep` (and
their mirror) and transforming back gives the envelope, the quefrencies
above `ncep` give the details. audioFlux uses a rectangular window by
default, so build the frames with `type=rect` to match it.
"""
function Cepstrogram(frames::Frames{T}; ncep::Int=4) where T
    N = get_winsize(frames)
    1 ≤ ncep < N ÷ 2 || throw(ArgumentError("ncep must be in 1:$(N ÷ 2 - 1), got $ncep"))
    nf = length(frames)
    half = N ÷ 2 + 1
    C = get_complex(Stft(frames))
    ceps = Matrix{T}(undef, half, nf)
    env  = Matrix{T}(undef, half, nf)
    det  = Matrix{T}(undef, half, nf)
    full = zeros(Complex{T}, N)
    y = Vector{T}(undef, N)
    lift = zeros(Complex{T}, N)
    pf = plan_fft!(copy(full)); pif = plan_ifft!(copy(full))
    for j in 1:nf
        # log power of the full (two-sided) spectrum
        @inbounds for k in 1:half
            full[k] = log(max(abs2(C[k, j]), T(1e-16)))
        end
        @inbounds for k in half+1:N
            full[k] = full[N - k + 2]
        end
        pif * full
        @inbounds for k in 1:N; y[k] = real(full[k]); end
        @inbounds for k in 1:half; ceps[k, j] = y[k]; end
        # envelope: quefrencies 0..ncep and their mirror (audioFlux)
        fill!(lift, zero(Complex{T}))
        @inbounds for k in 1:ncep+1; lift[k] = y[k]; end
        @inbounds for q in 0:ncep-1; lift[N - q] = y[q + 2]; end
        pf * lift
        @inbounds for k in 1:half; env[k, j] = real(lift[k]); end
        # details: quefrencies ncep+1 … N-ncep (audioFlux's range)
        fill!(lift, zero(Complex{T}))
        @inbounds for k in ncep+2:N-ncep+1; lift[k] = y[k]; end
        pf * lift
        @inbounds for k in 1:half; det[k, j] = real(lift[k]); end
    end
    return Cepstrogram{T}(ceps, env, det, frames, CepstrogramSetup(get_sr(frames), ncep, N))
end

# ---------------------------------------------------------------------------- #
#                           energy to zero-crossing ratio                      #
# ---------------------------------------------------------------------------- #
@descriptor Ezr TimeSetup """
    Ezr(frames::Frames; gamma=1) -> Ezr

Energy to zero-crossing ratio of every windowed frame (audioFlux
`Temporal.ezr`): `log10(1 + γ E) / (Z + 1)` with `E` the energy of the
windowed frame and `Z` its number of strict sign changes
(`x[i] x[i-1] < 0`).
"""
function Ezr(frames::Frames{T}; gamma::Real=1) where T
    g = T(gamma)
    v = _map_frames(frames; windowed=true) do x
        E = sum(abs2, x)
        Z = count(i -> x[i] * x[i - 1] < 0, 2:length(x))
        log10(1 + g * E) / (Z + 1)
    end
    return Ezr{typeof(frames),T}(v, frames, TimeSetup(get_sr(frames)))
end
