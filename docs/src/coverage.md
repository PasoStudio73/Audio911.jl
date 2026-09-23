```@meta
CurrentModule = Audio911
```

# [Feature coverage](@id coverage)

Inventory of the feature and utility functions of librosa (0.10) and of
MATLAB's Audio Toolbox (R2023b, `audioFeatureExtractor` features and the
standalone functions), with the Audio911 counterpart of each. "Parity"
means the output is checked numerically against a MATLAB `.mat` fixture;
"structural" means the algorithm follows the published definition and is
tested on synthetic signals and real audio but no reference output is
available offline.

## librosa

### `librosa.feature`

| librosa | Audio911 | status |
|:--------|:---------|:-------|
| `chroma_stft` | [`Chroma`](@ref), [`chroma_fbank`](@ref) | structural |
| `chroma_cqt`, `chroma_vqt` | [`Chroma`](@ref)`(::Cqt)` with [`cqt_chroma_fbank`](@ref) | parity with audioFlux `CQT.chroma` |
| `chroma_cens` | — | not implemented (CENS quantisation and smoothing) |
| `melspectrogram` | [`MelSpec`](@ref) (`scale=slaney, norm=bandwidth`) | parity (MATLAB fixtures); librosa settings structural |
| `mfcc` | [`Mfcc`](@ref), [`mfcc_librosa`](@ref) | structural |
| `rms` | [`Rms`](@ref) (frames or spectrogram) | structural |
| `spectral_centroid` | [`SpectralCentroid`](@ref) | parity |
| `spectral_bandwidth` | [`SpectralBandwidth`](@ref) | structural (`p=2` equals the parity-tested spread) |
| `spectral_contrast` | [`SpectralContrast`](@ref) | structural |
| `spectral_flatness` | [`SpectralFlatness`](@ref) | parity |
| `spectral_rolloff` | [`SpectralRolloff`](@ref) | parity |
| `poly_features` | [`PolyFeatures`](@ref) | structural |
| `tonnetz` | [`Tonnetz`](@ref) | structural |
| `zero_crossing_rate` | [`Zcr`](@ref) | structural |
| `tempogram` | [`Tempogram`](@ref) | structural |
| `fourier_tempogram`, `tempogram_ratio` | — | not implemented |
| `delta` | [`Delta`](@ref) (regression filter; librosa uses Savitzky–Golay) | parity with MATLAB, librosa filter not reproduced |
| `stack_memory` | — | not implemented (trivial delay embedding) |
| `inverse.mel_to_stft`, `mel_to_audio`, `mfcc_to_mel`, `mfcc_to_audio` | — | not implemented (inverse transforms) |

### `librosa.core`

| librosa | Audio911 | status |
|:--------|:---------|:-------|
| `load` | [`load`](@ref) | parity with MATLAB `audioread` |
| `stream` | — | not implemented (streaming) |
| `to_mono` | [`to_mono`](@ref) | done |
| `resample` | [`resample`](@ref) (polyphase FIR, `DSP.resample`) | done |
| `get_duration`, `get_samplerate` | [`get_duration`](@ref), [`get_samplerate`](@ref) | done |
| `stft` | [`Stft`](@ref) (power or magnitude; the complex STFT through [`get_complex`](@ref)) | parity |
| `istft` | [`istft`](@ref) (weighted or plain overlap-add) | parity with audioFlux |
| `griffinlim`, `phase_vocoder` | — | not implemented |
| `reassigned_spectrogram` | [`Reassign`](@ref), [`get_reassigned`](@ref) | parity with audioFlux `reassign` |
| `cqt`, `vqt` | [`Cqt`](@ref) (spectral-kernel CQT, `gamma` for the VQT) | parity with audioFlux (top octave exact) |
| `hybrid_cqt`, `pseudo_cqt`, `iirt` | [`Cqt`](@ref) covers the same representation; the hybrid/pseudo evaluation shortcuts and the IIR filterbank are not reproduced | partial |
| `magphase` | [`get_spec`](@ref) and [`get_phase`](@ref) | done |
| `fmt` | — | not implemented |
| `interp_harmonics`, `salience` | — | not implemented |
| `yin` | [`Pitch`](@ref) with [`pitch_yin`](@ref) | structural |
| `pyin` | — | not implemented (probabilistic YIN) |
| `piptrack`, `estimate_tuning`, `pitch_tuning` | — | not implemented |
| `zero_crossings` | [`zero_crossings`](@ref) | done |
| `autocorrelate` | [`autocorrelate`](@ref) | done |
| `lpc` | [`lpc`](@ref) (Burg) | structural |
| `A_weighting`, `C_weighting` | [`A_weighting`](@ref), [`C_weighting`](@ref) | done |
| `B_weighting`, `D_weighting`, `Z_weighting`, `frequency_weighting`, `multi_frequency_weighting` | — | not implemented |
| `perceptual_weighting` | [`perceptual_weighting`](@ref) | done |
| `amplitude_to_db`, `power_to_db`, `db_to_amplitude`, `db_to_power` | same names | done |
| `pcen` | [`pcen`](@ref) | structural |
| `mu_compress`, `mu_expand` | same names | done |
| `hz_to_mel`, `mel_to_hz`, `hz_to_midi`, `midi_to_hz`, `hz_to_note`, `note_to_hz`, `midi_to_note`, `note_to_midi`, `hz_to_octs` | same names | done |
| `hz_to_svara`, `midi_to_svara`, `hz_to_fjs`, key/interval helpers | — | not implemented |
| `fft_frequencies`, `mel_frequencies`, `cqt_frequencies`, `tempo_frequencies` | same names | done |
| `fourier_tempo_frequencies` | — | not implemented |
| `frames_to_samples`, `samples_to_frames`, `frames_to_time`, `time_to_frames`, `samples_to_time`, `time_to_samples` | same names (1-based) | done |
| `blocks_to_*` | — | not implemented |
| `clicks`, `tone`, `chirp` | same names | done |
| `get_fftlib`, `set_fftlib` | FFTW is used throughout | n/a |

### `librosa.effects`

| librosa | Audio911 | status |
|:--------|:---------|:-------|
| `hpss`, `harmonic`, `percussive` | [`Hpss`](@ref) on spectrograms (`decompose.hpss`); time-domain outputs need `istft` | partial |
| `time_stretch`, `pitch_shift` | — | not implemented |
| `remix` | — | not implemented |
| `trim`, `split` | [`trim_silence`](@ref), [`split_silence`](@ref) | done |
| `preemphasis`, `deemphasis` | [`preemphasis`](@ref), [`deemphasis`](@ref) | done |

### `librosa.filters`

| librosa | Audio911 | status |
|:--------|:---------|:-------|
| `mel` | [`auditory_fbank`](@ref) | parity |
| `chroma` | [`chroma_fbank`](@ref) | structural |
| `constant_q`, `wavelet`, `wavelet_lengths` | [`Cqt`](@ref) kernels ([`get_bandwidth`](@ref) gives the lengths), `Cwt` wavelets | partial |
| `semitone_filterbank`, `mr_frequencies` | — | not implemented (time-domain IIR filterbank) |
| `get_window` | DSP windows reexported plus [`povey`](@ref) | done |
| `window_bandwidth`, `window_sumsquare`, `diagonal_filter`, `cq_to_chroma` | — | not implemented |

### `librosa.onset` and `librosa.beat`

| librosa | Audio911 | status |
|:--------|:---------|:-------|
| `onset_strength`, `onset_strength_multi` | [`OnsetStrength`](@ref) (single channel) | structural |
| `onset_detect` | [`onset_detect`](@ref), [`peak_pick`](@ref) (librosa's exclusive window ends) | fixture through audioFlux's `Onset`, which uses the same peak picking |
| `onset_backtrack` | — | not implemented |
| `beat.tempo` | [`tempo`](@ref) | structural |
| `beat.beat_track` | [`beat_track`](@ref) | structural |
| `beat.plp` | — | not implemented |

### `librosa.decompose`, `librosa.segment`, `librosa.sequence`, `librosa.display`

| librosa | Audio911 | status |
|:--------|:---------|:-------|
| `decompose.hpss` | [`Hpss`](@ref) | structural |
| `decompose.decompose` (NMF), `nn_filter` | — | not implemented |
| `segment.*` (recurrence, agglomerative) | — | not implemented |
| `sequence.*` (DTW, Viterbi) | — | not implemented |
| `display.specshow`, `waveshow` | Plots recipes for every stage, see [Plotting](@ref plotting) | done |

## MATLAB Audio Toolbox

### `audioFeatureExtractor` features

| MATLAB | Audio911 | status |
|:-------|:---------|:-------|
| `linearSpectrum` | [`LinSpec`](@ref) | parity |
| `melSpectrum` | [`MelSpec`](@ref) | parity |
| `barkSpectrum` | [`BarkSpec`](@ref) | parity |
| `erbSpectrum` | [`ErbSpec`](@ref) | parity |
| `mfcc`, `mfccDelta`, `mfccDeltaDelta` | [`Mfcc`](@ref), [`Delta`](@ref) | parity |
| `gtcc`, `gtccDelta`, `gtccDeltaDelta` | [`Gtcc`](@ref), [`Delta`](@ref) | parity |
| `spectralCentroid`, `spectralCrest`, `spectralDecrease`, `spectralEntropy`, `spectralFlatness`, `spectralFlux`, `spectralKurtosis`, `spectralRolloffPoint`, `spectralSkewness`, `spectralSlope`, `spectralSpread` | `Spectral*` | parity (skewness structural) |
| `pitch` | [`Pitch`](@ref) (`NCF`, `CEP`, `PEF`, `LHS`, plus YIN, HPS and a spectral-peak method; `SRH` not implemented) | PEF/LHS parity with audioFlux, others structural |
| `harmonicRatio` | [`HarmonicRatio`](@ref) | structural |
| `zerocrossrate` | [`Zcr`](@ref) | structural |
| `shortTimeEnergy` | [`Energy`](@ref) | structural |

### Standalone functions

| MATLAB | Audio911 | status |
|:-------|:---------|:-------|
| `audioread`, `audioinfo` | [`load`](@ref), [`get_samplerate`](@ref) | parity |
| `audioresample` | [`resample`](@ref) | done |
| `designAuditoryFilterBank` | [`auditory_fbank`](@ref), [`gammatone_fbank`](@ref) | parity |
| `cepstralCoefficients` | [`Mfcc`](@ref) on any spectrogram | parity |
| `audioDelta` | [`Delta`](@ref) | parity |
| `melSpectrogram` | [`MelSpec`](@ref) | parity |
| `stft`, `istft` | [`Stft`](@ref), [`istft`](@ref) | done |
| `cwt` (Wavelet Toolbox) | [`Cwt`](@ref), [`cwt`](@ref) (`morse`, `morlet`, `bump`, `paul`, `dog`, `mexican`, `hermit`, `ricker`, L1 normalisation) | parity with audioFlux |
| `wavedec`, `wpdec`, `swt`, `modwt` (Wavelet Toolbox) | [`dwt`](@ref), [`wpt`](@ref), [`swt`](@ref) (periodic extension, 51 wavelets), [`Dwt`](@ref), [`Wpt`](@ref), [`Swt`](@ref) | parity with audioFlux |
| `wvd`, `emd`, `hht`, `ewt` (Signal Processing and Wavelet Toolboxes), `hilbert` | [`Wvd`](@ref), [`wvd`](@ref), [`Cwd`](@ref), [`emd`](@ref), [`Hht`](@ref), [`ewt`](@ref), [`Ewt`](@ref), [`hilbert`](@ref) | structural |
| `wsst`, `fsst` (Wavelet Toolbox, Signal Processing Toolbox) | [`Wsst`](@ref), [`wsst`](@ref); `fsst` is the STFT counterpart, see [`Reassign`](@ref) | parity with audioFlux |
| `noiseGate` | [`noisegate`](@ref); plus [`SpectralGate`](@ref) for a frequency range | structural |
| `detectSpeech`, `voiceActivityDetector` | — | not implemented |
| `integratedLoudness`, `loudnessMeter`, `splMeter` | — | not implemented |
| `pitch` | [`Pitch`](@ref) (`SRH` not implemented) | partial |
| `harmonicRatio`, `zerocrossrate` | [`HarmonicRatio`](@ref), [`Zcr`](@ref) | structural |
| `octaveFilterBank`, `gammatoneFilterBank` (time domain), `crossoverFilter` | — | not implemented (time-domain filter banks) |
| `shiftPitch`, `stretchAudio`, `audioTimeScaler` | — | not implemented |
| `kbdwin`, window functions | DSP windows, [`povey`](@ref) | partial |
| `vggishFeatures`, `openl3`, deep-learning helpers | — | out of scope |

## audioFlux

Every algorithm of audioFlux, with its counterpart, whether the definitions
agree and how each port is tested, is in the [audioFlux inventory](@ref audioflux).

## Not reached in this pass

Inverse transforms (`istft`, `griffinlim`, `mel_to_audio`), the
probabilistic pitch tracker (`pyin`), tuning estimation, time stretching
and pitch shifting, loudness meters, speech detection, NMF decomposition,
segmentation and sequence alignment are listed above as not implemented. They are omitted deliberately, not
silently: each needs either the complex STFT (which the pipeline does not
keep, to bound memory) or a substantial algorithm of its own.
