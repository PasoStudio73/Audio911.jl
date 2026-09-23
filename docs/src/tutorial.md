```@meta
CurrentModule = Audio911
```

# [Tutorial](@id tutorial)

This walkthrough builds a complete feature-extraction pipeline on the file
shipped with the tests, `test/test_files/test.wav` (16 kHz, mono, about
five seconds of speech), and shows where every knob lives.

## 1. Load the audio

```julia
using Audio911

path  = joinpath(pkgdir(Audio911), "test", "test_files", "test.wav")
audio = load(path; sr=8000, format=Float32, norm=true)
```

`load` reads WAV, FLAC, OGG and MP3. Say what you want explicitly: the
target sample rate (`sr`), the element type (`Float32` or `Float64`) and
whether to peak-normalise. Think about the sample rate first: if you only
care about 100–1000 Hz, analysing a 44.1 kHz file wastes most of the work;
resampling to 8 kHz makes everything after it several times cheaper.

Audio held in a matrix or a data-frame column is wrapped the same way:

```julia
x = rand(Float32, 16000)            # one second of samples
audio = AudioFile(x, 16000)         # mono, Float32, 16 kHz
```

## 2. Frames

Every transform starts by cutting the signal into overlapping frames and
attaching an analysis window.

```julia
frames = Frames(audio; winsize=256, winstep=128, type=hanning, periodic=true)
length(frames)          # number of frames
get_window(frames)      # the window, in the element type of the audio
get_times(Stft(frames)) # frame centres in seconds
```

`winsize` sets the frequency resolution (longer frames resolve closer
frequencies but smear time), `winstep` the hop. Start with 256 samples below
8 kHz and 512 above. Frames are lazy: nothing is copied until a transform
streams through them. Optional per-frame `preemph`, `dc_removal`,
`center` padding and `pad_end` reproduce what HTK, Kaldi, librosa and
python_speech_features do before their FFT.

## 3. From time to frequency: STFT or wavelets

```julia
stft = Stft(frames; nfft=512, spectrum=power)     # 257 bins × frames
cwt  = Cwt(frames; voices=12, freqrange=(50, 4000))  # scalogram on the same frames
```

`Stft` gives a uniform grid `sr/nfft` apart; `Cwt` a geometric grid with
`voices` scales per octave. Both are `AbstractSpectrogram`s and everything
below accepts either. Both can be built straight from the audio:

```julia
stft = Stft(audio; winsize=256, winstep=128, type=hanning, nfft=512)
```

Plot it:

```julia
using Plots
plot(stft; freq_scale=:log10)
```

## 4. Filterbanks and auditory spectrograms

The STFT is too detailed and linear in frequency. A filterbank groups its
bins into perceptual bands:

```julia
mel  = MelSpec(stft; nbands=26, scale=htk, norm=bandwidth, domain=:linear, freqrange=(100, 4000))
bark = BarkSpec(stft; nbands=24, freqrange=(100, 4000))
erb  = ErbSpec(stft; nbands=32, freqrange=(100, 4000))
lin  = LinSpec(stft; freqrange=(100, 4000), win_norm=true)
```

Design the filterbank yourself when you want to inspect or reuse it:

```julia
fb = auditory_fbank(stft; nbands=26, scale=slaney, norm=area)
plot(fb)
mel = MelSpec(stft, fb; win_norm=true)
```

A filterbank designed on one front end fits only that front end's grid;
`auditory_fbank(cwt; ...)` designs one for the scalogram.

## 5. Cepstra

```julia
mfcc = Mfcc(mel; ncoeffs=13, rect=mlog)   # MATLAB's log10 rectification
gtcc = Gtcc(erb; ncoeffs=13, rect=cubic_root)
d1   = Delta(mfcc)
d2   = Delta(d1)
get_data(mfcc)            # frames × 13
```

`Mfcc` exposes the rectification, floor, DCT scaling, lifter, first
coefficient and energy term because that is where the published MFCC
variants differ. If you need exactly what another toolkit computes, use
its preset instead of guessing:

```julia
k = mfcc_kaldi(audio)      # Kaldi compute-mfcc-feats defaults
l = mfcc_librosa(audio)    # librosa.feature.mfcc defaults
h = mfcc_htk(audio)        # HTK HCopy MFCC settings
```

The [MFCC variants](@ref mfcc_variants) page lists every axis and its source.

## 6. Descriptors and features

One value per frame:

```julia
c  = SpectralCentroid(lin)
sp = SpectralSpread(lin)
fl = SpectralFlux(lin; p=2)
rm = Rms(frames)
zc = Zcr(frames)
f0 = Pitch(frames; method=pitch_yin, range=(60, 400))
```

Matrices per frame:

```julia
ch  = Chroma(stft)                      # 12 pitch classes
tz  = Tonnetz(ch)                       # 6 tonal centroids
sc  = SpectralContrast(stft)            # 7 octave bands
on  = OnsetStrength(mel)                # onset envelope
idx = onset_detect(on)                  # onset frames
bpm = tempo(on)                         # global tempo
h   = Hpss(stft)                        # harmonic / percussive components
mel_h = MelSpec(get_harmonic(h); nbands=26)
```

Everything downstream carries the frame grid of the frames it came from:
`get_times(f0) == get_times(mfcc)` when both come from the same `Frames`.

## 7. Batch work

Audio911 stages are plain immutable objects, so mapping over files is a
`map`:

```julia
files = readdir("corpus"; join=true)
feats = map(files) do f
    a = load(f; sr=16000)
    get_data(Mfcc(MelSpec(Stft(a; winsize=512, winstep=256)); ncoeffs=13))
end
```

For a matrix whose columns are signals of the same sample rate:

```julia
feats = map(eachcol(M)) do x
    get_data(Mfcc(MelSpec(Stft(x, sr; winsize=512, winstep=256)); ncoeffs=13))
end
```

## 8. Keeping `Float32`

`load(...; format=Float32)` fixes the element type and nothing in the
pipeline promotes it: windows, filterbanks, DCT matrices, delta filters and
descriptors are all built in the audio's type. Use `Float64` when you want
to compare against MATLAB fixtures bit for bit; use `Float32` for size.
