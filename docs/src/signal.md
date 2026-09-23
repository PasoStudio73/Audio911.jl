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
  [`db_to_power`](@ref), [`db_to_amplitude`](@ref); weighting curves
  [`A_weighting`](@ref), [`C_weighting`](@ref), [`perceptual_weighting`](@ref).
- Companding and normalisation: [`mu_compress`](@ref), [`mu_expand`](@ref),
  [`normalize_signal`](@ref), [`normalize_peak`](@ref).
- Pre-emphasis: [`preemphasis`](@ref), [`deemphasis`](@ref),
  [`offset_compensation`](@ref).
- Silence: [`trim_silence`](@ref), [`split_silence`](@ref).
- Synthesis: [`tone`](@ref), [`chirp`](@ref), [`clicks`](@ref).
- Analysis: [`autocorrelate`](@ref), [`zero_crossings`](@ref), [`lpc`](@ref).
- Files: [`get_samplerate`](@ref), [`resample`](@ref), [`to_mono`](@ref).
