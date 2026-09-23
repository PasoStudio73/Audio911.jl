```@meta
CurrentModule = Audio911
```

# [Performance](@id performance)

Measured with `test/bench.jl` on the same machine (16 threads, Julia 1.13,
`--threads=auto --heap-size-hint=2G`), before the refactor (commit
`0e07a37`, the AudioReader/DataTreatments/Plots-based package) and after it.
Each stage runs twice and the second run is reported (`@timed`: wall time
and bytes allocated). "wav" is `test/test_files/test.wav` (16 kHz, 5.4 s);
"60s@44.1k" is sixty seconds of noise at 44.1 kHz (2.6 M samples), the
memory-relevant case.

| stage | before: time (s) | before: alloc (MiB) | after: time (s) | after: alloc (MiB) |
|:------|-----------------:|--------------------:|----------------:|-------------------:|
| load test.wav (Float32) | 0.000 | 0.5 | 0.000 | 0.4 |
| wav Stft (512/256 hamming) | 0.001 | 2.0 | 0.001 | 0.2 |
| wav MelSpec 26 bands | 0.000 | 0.1 | 0.000 | 0.1 |
| wav Mfcc 13 | 0.000 | 0.1 | 0.000 | 0.0 |
| wav Delta | 0.000 | 0.0 | 0.000 | 0.0 |
| wav ErbSpec 26 bands | 0.005 | 7.2 | 0.003 | 11.9 |
| wav Gtcc 13 | 0.000 | 0.1 | 0.000 | 0.0 |
| wav LinSpec | 0.000 | 0.1 | 0.000 | 0.1 |
| wav 11 spectral descriptors | 0.001 | 2.4 | 0.001 | 0.0 |
| 60s@44.1k Float32 Stft (512/256 hamming) | 0.054 | 152.8 | 0.002 | 10.2 |
| 60s@44.1k Float32 MelSpec 26 bands | 0.001 | 1.1 | 0.001 | 1.1 |
| 60s@44.1k Float32 Mfcc 13 | 0.002 | 6.2 | 0.003 | 1.5 |
| 60s@44.1k Float32 Delta | 0.000 | 1.0 | 0.001 | 0.5 |
| 60s@44.1k Float32 ErbSpec 26 bands | 0.007 | 29.2 | 0.006 | 12.9 |
| 60s@44.1k Float32 Gtcc 13 | 0.002 | 5.2 | 0.003 | 1.5 |
| 60s@44.1k Float32 LinSpec | 0.000 | 1.8 | 0.005 | 1.8 |
| 60s@44.1k Float32 11 spectral descriptors | 0.033 | 68.8 | 0.016 | 0.8 |
| 60s@44.1k Float64 Stft (512/256 hamming) | 0.096 | 304.2 | 0.003 | 20.4 |
| 60s@44.1k Float64 MelSpec 26 bands | 0.002 | 2.2 | 0.002 | 2.2 |
| 60s@44.1k Float64 Mfcc 13 | 0.002 | 5.2 | 0.004 | 3.1 |
| 60s@44.1k Float64 Delta | 0.000 | 1.0 | 0.001 | 1.0 |
| 60s@44.1k Float64 ErbSpec 26 bands | 0.005 | 9.0 | 0.006 | 13.8 |
| 60s@44.1k Float64 Gtcc 13 | 0.005 | 5.2 | 0.004 | 3.1 |
| 60s@44.1k Float64 LinSpec | 0.001 | 3.5 | 0.006 | 3.5 |
| 60s@44.1k Float64 11 spectral descriptors | 0.029 | 135.5 | 0.023 | 1.7 |

Peak resident set size of the whole benchmark process (which includes
loading the package and compiling the stages):

| | before | after |
|:--|------:|------:|
| peak RSS (MiB) | 1256 | 638 |

## What changed

- **Frames are lazy.** The old `Frames` copied every frame into a
  `winsize × nframes` matrix (twice the signal at 50 % overlap), then
  `Stft` windowed it into a second matrix, zero-padded into a third and
  ran a complex FFT over the whole thing. The new `Frames` stores the
  signal, the frame starts and the window; `Stft` streams through them
  with one `nfft` buffer per thread and a pre-planned real FFT, writing
  the power spectrum straight into the output. Allocation is now the
  output matrix; the 60 s Float32 STFT went from 153 MiB to 10 MiB.
- **Threads.** The STFT and the scalogram are threaded over frames and
  scales, the gammatone design over bands.
- **Descriptors work column by column** on the `bins × frames` matrix
  instead of building broadcast temporaries the size of the spectrogram;
  eleven descriptors on the 60 s spectrogram allocate 0.8 MiB instead of
  69 MiB.
- **Element type is preserved.** Windows, filterbanks, DCT matrices and
  delta filters are built in the audio's type, so a `Float32` pipeline
  allocates half of what a `Float64` one does instead of silently
  promoting.
- **Fewer dependencies.** Plots (now a Plots recipe, loaded only with
  Plots), DataTreatments (with DataFrames and friends) and AudioReader are
  gone; `using Audio911` loads FFTW, DSP, RecipesBase and the two codec
  libraries. This is most of the RSS and start-up difference.
- **No hidden mutation.** `Mfcc` used to floor the mel spectrogram in
  place; it now floors a copy.

## Regression checks

`test/allocations.jl` asserts allocation bounds for the hot stages on a
60 s `Float32` signal (a small multiple of the output size), so a future
change that reintroduces a full-size temporary fails the suite.
