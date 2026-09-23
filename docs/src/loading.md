```@meta
CurrentModule = Audio911
```

# [Loading audio](@id loading)

[`load`](@ref) opens WAV, FLAC and OGG (Vorbis) through libsndfile and MP3
through mpg123. The format is taken from the extension and verified against
the file's first bytes; a mismatch or an unsupported extension raises an
`ArgumentError`.

```julia
audio = load("speech.wav")                          # Float32, original rate, mono
audio = load("speech.wav"; sr=16000, format=Float64)
audio = load("music.mp3"; mono=false, norm=true)    # keep channels, peak-normalise
```

The result is an [`AudioFile`](@ref): a `frames × channels` matrix, the
sample rate after resampling, the original sample rate, the normalisation
flag and the path. Accessors: [`get_data`](@ref), [`get_sr`](@ref),
[`get_origin_sr`](@ref), [`get_nchannels`](@ref), [`is_norm`](@ref),
[`get_path`](@ref), [`get_duration`](@ref), `length`, `eltype`.

Processing happens in this order: element type conversion, mono
down-mix (channel average), resampling (polyphase FIR, `DSP.resample`),
peak normalisation. MP3 files are decoded to 16-bit PCM and scaled by
`1/32768`, which is what MATLAB's `audioread` returns; the WAV and MP3
readers are checked against `audioread` fixtures.

In-memory signals use the same type:

```julia
AudioFile(x, sr)                                    # vector or frames × channels matrix
AudioFile(x, sr; mono=false, norm=true, new_sr=8000, format=Float64)
```

Helpers: [`to_mono`](@ref), [`normalize_peak`](@ref), [`resample`](@ref),
[`get_samplerate`](@ref) (header only), [`detect_format`](@ref),
[`File`](@ref) and [`@format_str`](@ref) for explicit formats.
