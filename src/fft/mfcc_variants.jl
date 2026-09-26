# ---------------------------------------------------------------------------- #
#                         MFCC variants from the literature                    #
# ---------------------------------------------------------------------------- #
# Each preset is a plain composition of the public stages with the settings
# of a reference implementation; users can copy a preset and change any knob.
# See docs/src/mfcc_variants.md for the sources and the axes they differ on.

# ---------------------------------------------------------------------------- #
#                    integer-bin triangular filterbanks                        #
# ---------------------------------------------------------------------------- #
# ETSI ES 201 108 and python_speech_features define their triangles on
# integer FFT bins (rounded centre bins) instead of on continuous frequencies.

function _binfbank(::Type{T}, sr::Int, nfft::Int, nbands::Int, freqrange::FreqRange,
                   bin_of::Function, weight::Function) where {T<:AbstractFloat}
    edges = htk(Float64, freqrange, nbands)            # nbands + 2 edges in Hz
    cbin  = [bin_of(f) for f in edges]                # 0-based bin index of every edge
    nf    = _onesided_length(nfft)
    fb    = zeros(T, nbands, nf)
    for k in 1:nbands
        lo, c, hi = cbin[k], cbin[k + 1], cbin[k + 2]
        for i in lo:c
            0 ≤ i < nf && (fb[k, i + 1] = T(weight(i, lo, c, :rise)))
        end
        for i in c+1:hi
            0 ≤ i < nf && (fb[k, i + 1] = T(weight(i, c, hi, :fall)))
        end
    end
    sfreq = (0:nf-1) .* (T(sr) / T(nfft))
    freq  = T.(edges[2:end-1])
    bw    = T.(edges[3:end] .- edges[1:end-2])
    return FBank(fb, freq, bw, sr, nbands, :htk, none_norm, freqrange)
end

"""
    etsi_fbank(sr; nfft, nbands=23, freqrange=(64, sr÷2), T=Float64) -> FBank

Mel filterbank of ETSI ES 201 108 §4.2.6: centre bins `cbin = round(f·nfft/sr)`,
rising weight `(i - cbin_{k-1} + 1) / (cbin_k - cbin_{k-1} + 1)` and falling
weight `1 - (i - cbin_k) / (cbin_{k+1} - cbin_k + 1)`.
"""
function etsi_fbank(sr::Int; nfft::Int, nbands::Int=23, freqrange::FreqRange=(64, sr ÷ 2), T::Type=Float64)
    bin_of = f -> round(Int, f * nfft / sr)
    weight = (i, a, b, side) -> side == :rise ? (i - a + 1) / (b - a + 1) : 1 - (i - a) / (b - a + 1)
    return _binfbank(T, sr, nfft, nbands, freqrange, bin_of, weight)
end

"""
    psf_fbank(sr; nfft, nbands=26, freqrange=(0, sr÷2), T=Float64) -> FBank

Mel filterbank of python_speech_features `get_filterbanks`: edge bins
`floor((nfft + 1)·f/sr)`, weights `(i - bin_{k-1}) / (bin_k - bin_{k-1})` rising
and `(bin_{k+1} - i) / (bin_{k+1} - bin_k)` falling.
"""
function psf_fbank(sr::Int; nfft::Int, nbands::Int=26, freqrange::FreqRange=(0, sr ÷ 2), T::Type=Float64)
    bin_of = f -> floor(Int, (nfft + 1) * f / sr)
    weight = (i, a, b, side) -> b == a ? 0.0 : (side == :rise ? (i - a) / (b - a) : (b - i) / (b - a))
    return _binfbank(T, sr, nfft, nbands, freqrange, bin_of, weight)
end

# ---------------------------------------------------------------------------- #
#                               offset compensation                            #
# ---------------------------------------------------------------------------- #
"""
    offset_compensation(x; coef=0.999) -> Vector

ETSI ES 201 108 §4.2.2 DC-offset notch filter
`y[n] = x[n] - x[n-1] + coef * y[n-1]`.
"""
function offset_compensation(x::AbstractVector{T}; coef::Real=0.999) where {T<:Real}
    y = similar(x, float(T))
    k = float(T)(coef)
    xprev = zero(float(T)); yprev = zero(float(T))
    @inbounds for i in eachindex(x)
        yprev = x[i] - xprev + k * yprev
        xprev = x[i]
        y[i] = yprev
    end
    return y
end

# ---------------------------------------------------------------------------- #
#                                    presets                                   #
# ---------------------------------------------------------------------------- #
_ms(sr, ms) = round(Int, sr * ms / 1000)

"""
    mfcc_matlab(audio; kwargs...) -> Mfcc

MATLAB `audioFeatureExtractor` defaults: periodic Hamming window of 30 ms
with 20 ms overlap, power spectrum with window normalisation, 32 HTK-mel
bands normalised by bandwidth (linear domain), `log10` rectification,
orthonormal DCT, 13 coefficients, no lifter, no energy.

Keywords: `winsize`, `winstep`, `nfft`, `nbands`, `ncoeffs`, `rect`,
`freqrange`.
"""
function mfcc_matlab(audio::AudioFile;
    winsize::Int=_ms(get_sr(audio), 30), winstep::Int=winsize - _ms(get_sr(audio), 20),
    nfft::Int=winsize, nbands::Int=32, ncoeffs::Int=13, rect::Base.Callable=mlog,
    freqrange::FreqRange=(0, get_sr(audio) ÷ 2))
    frames = Frames(audio; winsize, winstep, type=hamming, periodic=true)
    stft   = Stft(frames; nfft, spectrum=power)
    mel    = MelSpec(stft; win_norm=true, nbands, scale=htk, norm=bandwidth, domain=:linear, freqrange)
    return Mfcc(mel; ncoeffs, rect, dct=dct_ortho)
end

"""
    mfcc_htk(audio; kwargs...) -> Mfcc

HTK `HCopy` with `TARGETKIND = MFCC_0` style settings (HTK Book §5.6):
25 ms frames every 10 ms, per-frame pre-emphasis 0.97, symmetric Hamming
window, magnitude spectrum, 20 triangular channels equally spaced on the mel
scale (warped domain, no normalisation), mel floor, natural log, DCT with
`sqrt(2/N)` on every row, 12 coefficients `c1..c12` (`first=1`), lifter 22.

- `scale=32768`: HTK works on 16-bit integer samples; normalised audio is
  scaled back so that the mel floor (`floor=1.0`) keeps its meaning
- `energy=false`: append the raw log energy (`_E`) as a last coefficient
- `c0=false`: also keep `c0` (`_0`) as the first of `ncoeffs + 1` coefficients

Keywords: `winsize`, `winstep`, `nfft`, `nbands`, `ncoeffs`, `preemph`,
`lifter`, `floor`, `freqrange`.
"""
function mfcc_htk(audio::AudioFile;
    winsize::Int=_ms(get_sr(audio), 25), winstep::Int=_ms(get_sr(audio), 10),
    nfft::Int=nextpow(2, winsize), nbands::Int=20, ncoeffs::Int=12, preemph::Real=0.97,
    lifter::Int=22, floor::Real=1.0, scale::Real=32768, energy::Bool=false, c0::Bool=false,
    freqrange::FreqRange=(0, get_sr(audio) ÷ 2))
    x      = get_data(audio) .* eltype(audio)(scale)
    frames = Frames(x, get_sr(audio); winsize, winstep, type=hamming, periodic=false, preemph)
    stft   = Stft(frames; nfft, spectrum=magnitude)
    mel    = MelSpec(stft; win_norm=false, nbands, scale=htk, norm=none_norm, domain=:warped, freqrange)
    return Mfcc(mel; ncoeffs=c0 ? ncoeffs + 1 : ncoeffs, first=c0 ? 0 : 1,
                rect=nlog, floor, dct=dct_htk, lifter,
                energy=energy ? raw_energy : nothing, energy_mode=:append)
end

"""
    mfcc_kaldi(audio; kwargs...) -> Mfcc

Kaldi `compute-mfcc-feats` defaults (`MfccOptions`, `FrameExtractionOptions`,
`MelBanksOptions`): 25 ms frames every 10 ms, DC removal and pre-emphasis
0.97 per frame, Povey window, FFT padded to the next power of two, power
spectrum, 23 mel bins from 20 Hz to Nyquist (triangles in the mel domain, no
normalisation), floor `eps(Float32)`, natural log, orthonormal DCT, 13
coefficients, lifter 22, and `C0` replaced by the raw log energy
(`use_energy=true`, `raw_energy=true`). Dithering is off.

Keywords: `winsize`, `winstep`, `nfft`, `nbands`, `ncoeffs`, `preemph`,
`lifter`, `freqrange`, `energy`.
"""
function mfcc_kaldi(audio::AudioFile;
    winsize::Int=_ms(get_sr(audio), 25), winstep::Int=_ms(get_sr(audio), 10),
    nfft::Int=nextpow(2, winsize), nbands::Int=23, ncoeffs::Int=13, preemph::Real=0.97,
    lifter::Int=22, energy::Bool=true, freqrange::FreqRange=(20, get_sr(audio) ÷ 2))
    frames = Frames(audio; winsize, winstep, type=povey, periodic=false, preemph, dc_removal=true)
    stft   = Stft(frames; nfft, spectrum=power)
    mel    = MelSpec(stft; win_norm=false, nbands, scale=htk, norm=none_norm, domain=:warped, freqrange)
    return Mfcc(mel; ncoeffs, rect=nlog, floor=eps(Float32), dct=dct_ortho, lifter,
                energy=energy ? raw_energy : nothing, energy_mode=:replace)
end

"""
    mfcc_librosa(audio; kwargs...) -> Mfcc

librosa `feature.mfcc` defaults: centred frames (constant padding) of
`n_fft=2048` every 512 samples, periodic Hann window, power spectrum, 128
Slaney-mel bands with Slaney (`2/bandwidth`) normalisation in the linear
domain, `power_to_db` (`10 log10`, `amin=1e-10`, `top_db=80`), orthonormal
DCT, 20 coefficients, no lifter (librosa's lifter, when set, starts its
index at 1: `lifter_offset=1`).

Keywords: `nfft`, `winsize`, `winstep`, `nbands`, `ncoeffs`, `lifter`,
`top_db`, `freqrange`, `pad_mode`.
"""
function mfcc_librosa(audio::AudioFile;
    nfft::Int=2048, winsize::Int=nfft, winstep::Int=nfft ÷ 4, nbands::Int=128, ncoeffs::Int=20,
    lifter::Int=0, top_db::Maybe{Real}=80, freqrange::FreqRange=(0, get_sr(audio) ÷ 2),
    pad_mode::Symbol=:constant)
    frames = Frames(audio; winsize, winstep, type=hanning, periodic=true, center=true, pad_mode)
    stft   = Stft(frames; nfft, spectrum=power)
    mel    = MelSpec(stft; win_norm=false, nbands, scale=slaney, norm=bandwidth, domain=:linear, freqrange)
    return Mfcc(mel; ncoeffs, rect=db, floor=1e-10, top_db, dct=dct_ortho, lifter, lifter_offset=1)
end

"""
    mfcc_etsi(audio; kwargs...) -> Mfcc

ETSI ES 201 108 v1.1.3 front end (Aurora): offset compensation notch filter,
25 ms frames every 10 ms, raw log energy (floored at -50), per-frame
pre-emphasis 0.97, symmetric Hamming window, FFT of 256 (8 kHz, 11 kHz) or
512 (16 kHz) points, magnitude spectrum, 23 integer-bin mel channels from
64 Hz to Nyquist ([`etsi_fbank`](@ref)), natural log with floor `2e-22`,
unscaled DCT, 13 coefficients `C0..C12` plus the log energy appended.

- `scale=32768`: the standard works on 16-bit integer samples.

Keywords: `winsize`, `winstep`, `nfft`, `nbands`, `ncoeffs`, `preemph`,
`freqrange`.
"""
function mfcc_etsi(audio::AudioFile;
    winsize::Int=_ms(get_sr(audio), 25), winstep::Int=_ms(get_sr(audio), 10),
    nfft::Int=nextpow(2, winsize), nbands::Int=23, ncoeffs::Int=13, preemph::Real=0.97,
    scale::Real=32768, freqrange::FreqRange=(64, get_sr(audio) ÷ 2))
    T      = eltype(audio)
    sr     = get_sr(audio)
    x      = offset_compensation(vec(get_data(audio)) .* T(scale))
    frames = Frames(x, sr; winsize, winstep, type=hamming, periodic=false, preemph)
    stft   = Stft(frames; nfft, spectrum=magnitude)
    fbank  = etsi_fbank(sr; nfft, nbands, freqrange, T)
    mel    = MelSpec(stft, fbank; win_norm=false)
    return Mfcc(mel; ncoeffs, first=0, rect=nlog, floor=2e-22, dct=dct_plain,
                energy=raw_energy, energy_mode=:append, energy_floor=exp(-50))
end

"""
    mfcc_psf(audio; kwargs...) -> Mfcc

python_speech_features `mfcc` defaults: signal-level pre-emphasis 0.97
(`y[1] = x[1]`), 25 ms frames every 10 ms with the last frame zero-padded
(`pad_end=true`), rectangular window, 512-point FFT, power spectrum scaled
by `1/nfft`, 26 integer-bin HTK-mel filters ([`psf_fbank`](@ref)), zeros
replaced by `eps`, natural log, orthonormal DCT, 13 coefficients, lifter 22,
and `C0` replaced by the log of the total spectral energy (`appendEnergy=True`).

Keywords: `winsize`, `winstep`, `nfft`, `nbands`, `ncoeffs`, `preemph`,
`lifter`, `freqrange`, `energy`.
"""
function mfcc_psf(audio::AudioFile;
    winsize::Int=_ms(get_sr(audio), 25), winstep::Int=_ms(get_sr(audio), 10),
    nfft::Int=512, nbands::Int=26, ncoeffs::Int=13, preemph::Real=0.97, lifter::Int=22,
    energy::Bool=true, freqrange::FreqRange=(0, get_sr(audio) ÷ 2))
    T      = eltype(audio)
    sr     = get_sr(audio)
    x      = preemphasis(vec(get_data(audio)); coef=preemph, zi=0)
    frames = Frames(x, sr; winsize, winstep, type=rect, periodic=false, pad_end=true)
    stft   = Stft(frames; nfft, spectrum=power, scale=1 / nfft)
    fbank  = psf_fbank(sr; nfft, nbands, freqrange, T)
    mel    = MelSpec(stft, fbank; win_norm=false)
    return Mfcc(mel; ncoeffs, rect=nlog, floor=eps(Float64), dct=dct_ortho, lifter,
                energy=energy ? spectrum_energy : nothing, energy_mode=:replace, energy_floor=eps(Float64))
end
