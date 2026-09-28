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
audio = load("music.mp3"; norm=true)                # stereo is averaged to mono, peak-normalise
```

The result is an [`AudioFile`](@ref): a mono `Vector` of samples, the
sample rate after resampling, the original sample rate, the normalisation
flag and the path. Multi-channel files are averaged to mono once, right
after decoding; every stage after that works on mono vectors only.
Accessors: [`get_data`](@ref), [`get_sr`](@ref),
[`get_origin_sr`](@ref), [`is_norm`](@ref),
[`get_path`](@ref), [`get_duration`](@ref), `length`, `eltype`.

Processing happens in this order: element type conversion, mono
down-mix (channel average), resampling (polyphase FIR, `DSP.resample`),
peak normalisation. MP3 files are decoded to 16-bit PCM and scaled by
`1/32768`, which is what MATLAB's `audioread` returns; the WAV and MP3
readers are checked against `audioread` fixtures.

In-memory signals use the same type:

```julia
AudioFile(x, sr)                                    # a mono vector
AudioFile(x, sr; norm=true, new_sr=8000)
AudioFile(vec(to_mono(X)), sr)                      # a frames × channels matrix, averaged first
```

Helpers: [`to_mono`](@ref), [`normalize_peak`](@ref), [`resample`](@ref),
[`get_samplerate`](@ref) (header only), [`detect_format`](@ref),
[`File`](@ref) and [`@format_str`](@ref) for explicit formats.
