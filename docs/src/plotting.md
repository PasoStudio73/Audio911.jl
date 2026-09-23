```@meta
CurrentModule = Audio911
```

# [Plotting](@id plotting)

Audio911 defines [Plots](https://docs.juliaplots.org) recipes for every
stage through RecipesBase, so plotting costs nothing until you load Plots:

```julia
using Audio911, Plots

plot(audio)                         # waveform
plot(frames)                        # signal with the window on the first frame
plot(stft)                          # dB heatmap, time × frequency
plot(mel; freq_scale=:log10)        # log frequency axis
plot(cwt; db=false)                 # raw values
plot(fbank)                         # filter curves
plot(mfcc); plot(delta)             # coefficients × time
plot(SpectralCentroid(lin))         # one line against time
plot(Chroma(stft)); plot(Tonnetz(stft))
plot(Hpss(stft))                    # harmonic and percussive panels
```

Spectrogram recipes accept `db` (default `true`), `top_db` (clip below the
maximum, default 80) and `freq_scale` (`:linear` or `:log10`). Every recipe
returns an ordinary Plots object, so the usual attributes (`size`, `c`,
`clims`, `title`, ...) apply, and `plot!` overlays work.
