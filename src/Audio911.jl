module Audio911
using  Reexport

# ---------------------------------------------------------------------------- #
#                           audio related packages                             #
# ---------------------------------------------------------------------------- #
using  FFTW
import DSP
using  LinearAlgebra
using  Statistics: mean, std, median
using  Printf: @sprintf
using  RecipesBase

# codecs for the internal audio loader
using  libsndfile_jll: libsndfile
using  mpg123_jll: libmpg123

# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
"""
    AbstractAudioFile

Supertype of loaded audio ([`AudioFile`](@ref)).
"""
abstract type AbstractAudioFile end

"""
    AbstractSetup

Supertype of the immutable `info` records every stage carries.
"""
abstract type AbstractSetup end

"""
    AbstractFrame

Supertype of framed time-domain signals ([`Frames`](@ref)).
"""
abstract type AbstractFrame end

"""
    AbstractFBank

Supertype of filterbank designs ([`FBank`](@ref)).
"""
abstract type AbstractFBank end

"""
    AbstractAudioSpectrum

Supertype of every frame-rate feature of the pipeline.
"""
abstract type AbstractAudioSpectrum end

"""
    AbstractSpectrogram <: AbstractAudioSpectrum

A time-frequency representation: a real, non-negative `bins × frames` matrix
with a frequency axis. Front ends (`Stft`, `Cwt`) and filterbank outputs
(`LinSpec`, `MelSpec`, `BarkSpec`, `ErbSpec`) are subtypes. See the
[pipeline design](@ref design) for the interface a subtype must implement.
"""
abstract type AbstractSpectrogram <: AbstractAudioSpectrum end

"""
    AbstractCepstrum <: AbstractAudioSpectrum

Cepstral coefficients (`Mfcc`, `Gtcc`).
"""
abstract type AbstractCepstrum <: AbstractAudioSpectrum end

"""
    AbstractDelta <: AbstractAudioSpectrum

Temporal derivatives of a feature matrix (`Delta`).
"""
abstract type AbstractDelta <: AbstractAudioSpectrum end

"""
    AbstractSpectral <: AbstractAudioSpectrum

One value per frame descriptors (`SpectralCentroid`, `Rms`, ...).
"""
abstract type AbstractSpectral <: AbstractAudioSpectrum end

# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
# type alias for `Union{T, Nothing}`
const Maybe{T} = Union{T, Nothing}

"""
    AudioData

Type alias for audio sample data: `Float64` or `Float32`. Every stage of the
pipeline is parameterised on one of these two types and never promotes one to
the other.
"""
const  AudioData = Union{Float64, Float32}
export AudioData

"""
    FreqRange

A frequency range in Hz, as a tuple `(min, max)` of integers.

```julia
fr::FreqRange = (20, 20000)
```
"""
const  FreqRange = Tuple{T, T} where {T<:Int64}

"""
    get_low(r), get_hi(r)

Lower and upper bound of a [`FreqRange`](@ref) or [`ScaleRange`](@ref).
"""
get_low(r::FreqRange) = r[1]

"""
    get_hi(r)

Upper bound of a [`FreqRange`](@ref) or [`ScaleRange`](@ref); see [`get_low`](@ref).
"""
get_hi(r::FreqRange)  = r[2]
export FreqRange, get_low, get_hi

"""
    ScaleRange

A range on a perceptual scale (mel, bark, ERB), as a tuple of `AudioData`.
"""
const  ScaleRange  = Tuple{T, T} where {T<:AudioData}

get_low(r::ScaleRange) = r[1]
get_hi(r::ScaleRange)  = r[2]
export ScaleRange

# ---------------------------------------------------------------------------- #
#                           spectrum normalizations                            #
# ---------------------------------------------------------------------------- #
"""
    winpower(f, w)

Window normalisation for a power spectrum: `f / sum(w)^2`.
"""
winpower(f, w)     = f / sum(w)^2

"""
    winmagnitude(f, w)

Window normalisation for a magnitude spectrum: `f / sum(w)`.
"""
winmagnitude(f, w) = f / sum(w)

export winpower, winmagnitude

# ---------------------------------------------------------------------------- #
#                                  modules                                     #
# ---------------------------------------------------------------------------- #
# reexport DSP's window functions
@reexport using DSP: rect, hanning, hamming, cosine, lanczos, triang
@reexport using DSP: bartlett, bartlett_hann, blackman

# accessor generics shared by audio files and every stage
function get_data end
function get_sr end

export @format_str, File, AudioFormat, AudioFile, load
export filename, file_extension, formatname, detect_format
export get_origin_sr, get_nchannels, is_norm, get_path
export to_mono, normalize_peak
include("audio/formats.jl")
include("audio/libsndfile.jl")
include("audio/mpg123.jl")
include("audio/audiofile.jl")

export get_spec, get_spectrum, get_window, get_winnorm
export get_nbins, get_nframes, get_times, get_offset, get_duration
include("interface.jl")

export Frames, povey, movingwindow
export preemphasis, deemphasis
include("frames.jl")

export Stft
export power, magnitude
include("fft/stft.jl")

export Cwt
export morlet, morse, bump
include("wavelet/cwt.jl")

export FBank
export htk, slaney, bark
export area, bandwidth, none_norm
export auditory_fbank, gammatone_fbank
include("fft/fbank.jl")

export LinSpec
include("fft/lin_spec.jl")

export MelSpec, BarkSpec, ErbSpec
include("fft/mel_spec.jl")

export Mfcc, Gtcc
export mlog, nlog, cubic_root, db
export dct_ortho, dct_htk, dct_plain
include("fft/mfcc.jl")

export mfcc_matlab, mfcc_htk, mfcc_kaldi, mfcc_librosa, mfcc_etsi, mfcc_psf
export etsi_fbank, psf_fbank, offset_compensation
include("fft/mfcc_variants.jl")

export Delta
include("fft/delta.jl")

export SpectralCentroid, SpectralCrest, SpectralDecrease, SpectralEntropy
export SpectralFlatness, SpectralFlux, SpectralKurtosis, SpectralRolloff
export SpectralSkewness, SpectralSlope, SpectralSpread, SpectralBandwidth
include("fft/spectral.jl")

# signal utilities: conversions, decibels, weighting, synthesis, trim, lpc
export hz_to_mel, mel_to_hz, hz_to_midi, midi_to_hz, midi_to_note, note_to_midi, hz_to_note, note_to_hz
export fft_frequencies, mel_frequencies, cqt_frequencies, tempo_frequencies
export frames_to_samples, samples_to_frames, frames_to_time, time_to_frames, samples_to_time, time_to_samples
export power_to_db, amplitude_to_db, db_to_power, db_to_amplitude
export A_weighting, C_weighting, perceptual_weighting
export mu_compress, mu_expand, normalize_signal
export tone, chirp, clicks, trim_silence, split_silence, lpc, get_samplerate
include("signal/utils.jl")

# frame-level and derived features
export Rms, Energy, Zcr, Pitch, HarmonicRatio
export zero_crossings, autocorrelate, pitch_ncf, pitch_yin, pitch_cep
include("features/timedomain.jl")

export Chroma, ChromaFBank, chroma_fbank, hz_to_octs, Tonnetz, SpectralContrast, PolyFeatures
include("features/chroma.jl")

export OnsetStrength, peak_pick, onset_detect, Tempogram, tempo, beat_track
include("features/onset.jl")

export DerivedSpec, Hpss, get_harmonic, get_percussive, get_masks, get_name
export noisegate, SpectralGate, pcen
include("features/hpss.jl")

# ---------------------------------------------------------------------------- #
#                                  methods                                     #
# ---------------------------------------------------------------------------- #
# general
export get_data, get_setup

# stft related
export get_size, get_step, get_overlap
export get_window, get_winframes, get_winsize, get_energy

# spectrogram related
export get_freq, get_sr, get_nfft, get_spectrum

# filterbank related
export get_bandwidth
export get_nbands, get_scale, get_norm
export get_freqrange

# cepstrum related
export get_ncoeffs, raw_energy, spectrum_energy
export get_fbank, get_frames, get_parent, get_frontend, get_signal, frame!, get_scales

# ---------------------------------------------------------------------------- #
#                                   plots                                      #
# ---------------------------------------------------------------------------- #
# Plots recipes: `using Plots; plot(x)` works for every stage, at no cost
# when Plots is not loaded.
include("plots.jl")

end
