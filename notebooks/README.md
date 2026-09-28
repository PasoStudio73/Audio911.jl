# Audio911 interactive tour

Nine [Pluto](https://plutojl.org) notebooks that demonstrate every algorithm in Audio911.
Each one loads a signal (the speech file from the test suite or one of three synthetic signals), runs the algorithms with interactive controls for every parameter, and explains in text how each parameter changes the result.

| notebook | contents |
|:--|:--|
| [01_loading_and_frames](01_loading_and_frames.jl) | `load`, `AudioFile`, formats, `Frames` and every framing option, window functions, pre-emphasis |
| [02_stft_and_filterbanks](02_stft_and_filterbanks.jl) | `Stft`, `istft`, `FBank` scales / norms / shapes, `LinSpec`, `MelSpec`, `BarkSpec`, `ErbSpec` |
| [03_cepstra](03_cepstra.jl) | `Mfcc`, `Gtcc`, `Delta`, the MATLAB / HTK / Kaldi / librosa / ETSI / python_speech_features MFCC variants, cepstrogram and deconvolution |
| [04_spectral_descriptors](04_spectral_descriptors.jl) | every one-value-per-frame spectral descriptor, spectral contrast, poly features |
| [05_time_frequency](05_time_frequency.jl) | `Cwt` and its wavelets, `Cqt`, `Pwt`, `St` / `Fst`, `Nsgt`, synchrosqueezing, reassignment, Wigner-Ville and Cohen class |
| [06_discrete_wavelets_and_decompositions](06_discrete_wavelets_and_decompositions.jl) | `Dwt`, `Wpt`, `Swt`, EMD / Hilbert-Huang, empirical wavelet transform |
| [07_music_and_rhythm](07_music_and_rhythm.jl) | chroma, tonnetz, onset strength, novelty, tempo, beat tracking, HPSS, gating, PCEN, harmonic count |
| [08_time_domain_and_pitch](08_time_domain_and_pitch.jl) | RMS, energy, zero-crossing rate, harmonic ratio, every pitch estimator |
| [09_signal_processing_and_classic](09_signal_processing_and_classic.jl) | unit conversions, weighting curves, LPC, Hilbert / CZT / correlation, time stretch and pitch shift, NMF, HMM / Viterbi |

## Running

From the repository root:

```bash
test/run.sh pluto        # starts Pluto and opens notebook 01; open the others from Pluto's file picker
```

or, from any Julia session with Pluto installed:

```julia
using Pluto
Pluto.run(notebook="notebooks/01_loading_and_frames.jl")
```

Each notebook activates the environment in this folder (`Project.toml`), which takes Audio911 from this repository plus PlutoUI and Plots.
The first run instantiates that environment and precompiles Plots, which takes a minute or two.
`common.jl` holds the signal menu and the in-browser audio player shared by all notebooks.

## Checking

```bash
test/run.sh notebooks                           # run every notebook headless
test/run.sh notebooks notebooks/03_cepstra.jl   # or just one
```

This runs each notebook with its default control values and lists every cell that throws, so a library change that breaks the tour shows up without opening a browser.
