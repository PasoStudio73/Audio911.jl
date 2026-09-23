```@meta
CurrentModule = Audio911
```

# [Signal utilities](@id signal)

Small functions that librosa and MATLAB users expect around the pipeline.

- Scales: [`hz_to_mel`](@ref), [`mel_to_hz`](@ref), [`hz_to_midi`](@ref),
  [`midi_to_hz`](@ref), [`hz_to_note`](@ref), [`note_to_hz`](@ref),
  [`midi_to_note`](@ref), [`note_to_midi`](@ref), [`hz_to_octs`](@ref).
- Grids: [`fft_frequencies`](@ref), [`mel_frequencies`](@ref),
  [`cqt_frequencies`](@ref), [`tempo_frequencies`](@ref).
- Frames, samples, seconds: [`frames_to_samples`](@ref),
  [`samples_to_frames`](@ref), [`frames_to_time`](@ref),
  [`time_to_frames`](@ref), [`samples_to_time`](@ref),
  [`time_to_samples`](@ref) (all 1-based).
- Decibels: [`power_to_db`](@ref), [`amplitude_to_db`](@ref),
  [`db_to_power`](@ref), [`db_to_amplitude`](@ref) (`min_db` floors the
  result, as audioFlux's absolute dB); weighting curves [`A_weighting`](@ref),
  [`B_weighting`](@ref), [`C_weighting`](@ref), [`D_weighting`](@ref),
  [`perceptual_weighting`](@ref).
- Companding and normalisation: [`mu_compress`](@ref), [`mu_expand`](@ref),
  [`normalize_signal`](@ref), [`normalize_peak`](@ref).
- Pre-emphasis: [`preemphasis`](@ref), [`deemphasis`](@ref),
  [`offset_compensation`](@ref).
- Silence: [`trim_silence`](@ref), [`split_silence`](@ref).
- Synthesis: [`tone`](@ref), [`chirp`](@ref), [`clicks`](@ref), [`synth_f0`](@ref)
  (a pitch curve).
- Analysis: [`autocorrelate`](@ref), [`xcorr`](@ref), [`convolve`](@ref),
  [`czt`](@ref) (chirp-Z and zoom FFT), [`zero_crossings`](@ref), [`lpc`](@ref).
- Feature scaling and levels: [`feature_scale`](@ref) (audioFlux's seven
  scalers), [`temporal_db`](@ref).
- Files: [`get_samplerate`](@ref), [`resample`](@ref) (polyphase FIR, or
  `method=:sinc` for audioFlux's band-limited sinc interpolation with its
  `:best`, `:mid` and `:fast` Kaiser filters), [`to_mono`](@ref).
- Hilbert transform: [`hilbert`](@ref).

## Time and pitch modification

[`time_stretch`](@ref) changes the duration of a signal by `1 / rate`
through the [`phase_vocoder`](@ref) and the overlap-add [`istft`](@ref);
[`pitch_shift`](@ref) moves the pitch by a number of semitones, a time
stretch followed by band-limited resampling. Both are ports of audioFlux's
`TimeStretch` and `PitchShift` and agree with them to float32 precision
(`test/stretch_af.jl`); they take vectors or an [`AudioFile`](@ref).

```julia
slow   = time_stretch(x, 0.5)                  # twice as long, same pitch
up     = pitch_shift(x, 3)                     # three semitones up, same length
audio2 = pitch_shift(audio, -2)                # every channel of an AudioFile
y8k    = Audio911.resample(x, 16000, 8000; method=:sinc, quality=:best)
C2     = phase_vocoder(get_complex(stft), 1.5; hop=get_step(stft))
```
