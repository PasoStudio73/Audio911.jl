# Audio911.jl

Julia package for audio feature extraction (STFT/wavelet front ends, mel/bark/ERB
spectrograms, MFCC/GTCC in every published variant, descriptors, chroma, onsets,
HPSS) with numerical parity against MATLAB's Audio Toolbox as the test oracle.

## Commands

Use `test/run.sh` for every Julia invocation; it passes `--startup-file=no`
(the machine-wide `~/.julia/config/startup.jl` activates an unrelated project),
`--project=<repo>`, a 2G heap hint and a 4G systemd memory cap.

- `test/run.sh test` - full suite (`Pkg.test`, MAT and Plots come from the test extras)
- `test/run.sh docs` - Documenter build into `docs/build`
- `test/run.sh bench` - `test/bench.jl`, the table in `docs/src/performance.md`
- `test/run.sh file X.jl` - one test file in a temporary environment

`test/runtests.jl` is a hand-maintained list; a new test file runs only once
it is added there.

## Architecture

Read `docs/src/design.md` first: it defines the front-end interface
(`get_spec` in bins × frames, `get_freq`, `get_sr`, `get_spectrum`,
`get_window`/`get_winnorm`, `get_times`) that every downstream stage
dispatches on. `Stft` and `Cwt` are the front ends; `LinSpec`, `MelSpec`,
`BarkSpec`, `ErbSpec`, `Mfcc`, `Delta`, the `Spectral*` descriptors and the
features in `src/features/` consume the interface, never a concrete type.

- `get_data` keeps the public orientation per type (`Stft`/`FBank` bins × frames,
  filterbank spectrograms and cepstra frames × bands); new code uses `get_spec`.
- Every stage keeps a `parent` reference (`get_parent`, `get_frames`,
  `get_frontend`) and inherits the time grid from its `Frames`.
- Options are functions compared by identity (`hamming`, `power`, `htk`,
  `bandwidth`, `mlog`, `dct_ortho`, `raw_energy`); the Symbol exceptions are
  `domain=:linear|:warped`, `energy_mode=:replace|:append|:prepend` and the
  mode switches of the audioFlux ports (`mode`, `accumulate`, `method`).
- Element type flows from `load(...; format=Float32|Float64)`; nothing may
  promote it (`test/pipeline.jl` "Float32 stays Float32").
- Audio loading is internal (`src/audio/`, libsndfile_jll + mpg123_jll).
- Plotting is RecipesBase recipes in `src/plots.jl`; Plots is only a test extra.

## Test contract

Every `isapprox(..., mat["features"])` against `test/matlab_files/**/*.mat` is a
fixed contract; the MATLAB calls that produced each fixture are in
`test/matlab_sources/*.m` and repeated as comments in the Julia tests. Changing
a feature means adding a `.m` recipe and a `.mat` fixture, not relaxing a test.
`test/allocations.jl` bounds allocations of the hot stages on a 60 s signal.

## Docs

`docs/make.jl` lists the pages; `docs/src/api.md` is generated from
docstrings, so every exported symbol needs one (the build warns otherwise).
`docs/src/coverage.md` records what of librosa and MATLAB is implemented and
what is deliberately not; update it when adding or dropping a feature.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
