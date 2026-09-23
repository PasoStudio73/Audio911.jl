```@meta
CurrentModule = Audio911
```

# [audioFlux inventory](@id audioflux)

Inventory of every algorithm of [audioFlux](https://github.com/libAudioFlux/audioFlux)
(MIT licence, snapshot of the `master` branch, Python package 0.1.9) with its
Audio911 counterpart. The definitions were read from the C sources under
`src/` (the ground truth), the headers under `include/` and the Python
wrappers under `python/audioflux/`.

Columns:

- **Audio911**: the counterpart stage or function, or `—`.
- **definition**: whether the existing counterpart computes the same thing as
  audioFlux, and how it differs when it does not.
- **status**: `port` (new in this expansion), `already covered`,
  `extend existing` (a keyword or variant added to an existing stage) or
  `out of scope` with the reason.
- **oracle**: how the port is tested. `fixture` means the output is compared
  numerically with audioFlux 0.1.9 run on `test/test_files/test.wav` (the
  generating scripts are in `test/audioflux_sources/`, the `.mat` files in
  `test/audioflux_files/`, see `test/run.sh oracle`); `structural` means the
  algorithm is tested on synthetic signals and real audio against its
  published definition, because audioFlux exposes no reference output for it
  (no Python wrapper, header-only declaration, or an output format that is not
  comparable); `MATLAB` means the existing MATLAB parity fixture stays the
  oracle.

Conventions that differ globally and are not repeated in every row:

- audioFlux works in `float32` on signals whose length is a power of two
  (`radix2_exp`) for the whole-signal transforms; Audio911 works in the element
  type of `load` on any length. Fixtures are generated on the first
  `2^k` samples where audioFlux requires it and compared with a `float32`
  tolerance.
- audioFlux returns complex matrices for the transforms; Audio911 front ends
  store a real `power` or `magnitude` spectrogram (see
  [the complex-spectrum accessor](@ref design_complex)) and pool
  whole-signal transforms onto the `Frames` grid, so they plug into the same
  downstream stages as `Stft`.
- audioFlux's default sample rate is 32 kHz and its default frame is
  `radix2_exp=12` (4096 samples) with a hop of 1024; Audio911 takes both from
  `Frames`.

## Transforms

| audioFlux | Audio911 | definition | status | oracle |
|:----------|:---------|:-----------|:-------|:-------|
| `BFT` (linear scale) | [`Stft`](@ref) | same one-sided STFT; audioFlux's `power` is `|X|²` without window normalisation, same as `Stft`. audioFlux's `is_continue` (state across calls) has no counterpart | already covered | MATLAB |
| `BFT` (linspace/mel/bark/erb/octave/log scales) | [`MelSpec`](@ref), [`BarkSpec`](@ref), [`ErbSpec`](@ref), [`LinSpec`](@ref) on an `Stft` | a BFT with a scale is an STFT times a filterbank: the same pipeline as Audio911's `Stft → MelSpec`. Scales `linspace`, `erb` (triangular, Glasberg–Moore ERB-rate), `octave` and `log` and the window filter styles are new, see the scale and style sections | extend existing (`auditory_fbank` scales and styles) | fixture |
| `BFT(is_reassign=True)` | [`Reassign`](@ref) | see reassignment below | port | fixture |
| `BFT(is_temporal=True)` (energy, rms, zcr per frame) | [`Energy`](@ref), [`Rms`](@ref), [`Zcr`](@ref) | same three per-frame values | already covered | structural |
| `NSGT` (efficient and standard filter banks) | — | non-stationary Gabor transform: the whole-signal FFT is windowed band by band (window length from the band edges, `min_len` floor) and inverse-transformed at the band's own length | port as [`Nsgt`](@ref); the per-band time resolution is pooled onto the `Frames` grid | structural (audioFlux returns a ragged cell array resampled to a common length, not comparable bin by bin; band centre frequencies and band energies are checked against the fixture) |
| `CWT` (morse, morlet, bump, paul, dog, mexican, hermit, ricker; all scale types; `is_padding`) | [`Cwt`](@ref), [`cwt`](@ref) | `morse` and `bump` are identical (peak normalised to 2). audioFlux's `morlet` is `2 exp(-(ω-ω0)²/β)` with `β=2`, Audio911's is `π^(-1/4) exp(-(ω-ω0)²/2)`: same shape, constant factor differs (kept, documented). audioFlux computes the complex transform on the whole signal; Audio911's `Cwt` pools `|W|²` over frames. audioFlux's `linear` grid is the `linspace` grid on FFT-bin frequencies | extend existing: wavelets [`paul`](@ref), [`dog`](@ref), [`mexican`](@ref), [`hermit`](@ref), [`ricker`](@ref); `scale`/`nbands`/`bins_per_octave` keywords for the octave/mel/bark/erb/linspace/log grids; whole-signal [`cwt`](@ref) with `pad`; `get_complex` | fixture (every wavelet with and without padding, every grid) |
| `PWT` (pseudo wavelet transform) | — | the whole-signal FFT multiplied by window-shaped bandpass filters designed on any scale, one inverse FFT per band | port as [`Pwt`](@ref) | fixture |
| `CQT` | — (the coverage table pointed to `Cwt` with `bump` as a constant-Q substitute) | Brown–Puckette spectral kernels: per bin a windowed complex exponential of length `Q·sr/f`, FFT'd, thresholded at `thresh` (0.01), applied to the frame FFT; `factor` scales Q, `norm` none/area/bandwidth, `is_scale` divides by the kernel length | port as [`Cqt`](@ref) | fixture |
| `VQT` | — | the CQT with `beta > 0` (audioFlux's `beta` is the γ of Schörkhuber et al. 2014): kernel length `Q·sr/(f + β/(2^(1/b)-1))`; audioFlux applies it in the top octave only (the decimated octaves reuse the top octave's lengths), Audio911 in every bin | port: `Cqt(...; gamma=...)` | fixture (top octave) |
| `ST` (S-transform, `factor`, `norm`) | — | Stockwell transform: `ifft(X(f+k) · G_k(f))` with `G_k(f) = exp(-2π²λ f²/k^(2p))` plus its mirror, per frequency index `k` | port as [`St`](@ref) | fixture |
| `FST` (fast S-transform) | — | Brown, Lauzon & Frayne 2010 dyadic sampling: FFT, shifted dyadic partition, one inverse FFT per partition, index map to the `(N/2+1) × N` image | port as [`fst`](@ref) (whole signal, exact) and [`Fst`](@ref) (pooled front end) | fixture |
| `DWT` (`num` levels, wavelet types) | — | Mallat cascade with the wavelet's analysis filters and periodic extension; returns the concatenated coefficients and a `levels × N` image with every level's coefficients repeated to full length. audioFlux 0.1.9's Python `DWT` passes an extra `samplate` argument to `dwtObj_new`, so it always applies `sym4`; the fixtures call the C library directly | port as [`dwt`](@ref)/[`Dwt`](@ref) | fixture (all 51 wavelets) |
| `WPT` (wave packet transform) | — | full binary tree of `num` levels, `2^num` bands | port as [`wpt`](@ref)/[`Wpt`](@ref) | fixture |
| `SWT` (stationary wavelet transform) | — | undecimated à-trous cascade, approximation and detail per level | port as [`swt`](@ref)/[`Swt`](@ref) | fixture |
| discrete wavelet families `haar`, `db2–10/20/30/40`, `sym2–10/20/30`, `coif1–5`, `fk4/6/8/14/18/22`, `bior1.1–6.8`, `dmey` | — | filter coefficients tabulated in `src/filterbank/coef/` (six decimals) | port: [`wavelet_filters`](@ref), [`DISCRETE_WAVELETS`](@ref); the table is generated from audioFlux's headers by `test/audioflux_sources/_gen_dwt_coefs.py` | fixture (every wavelet through `dwt`) |
| `reassign` (STFT reassignment, `re_type` all/fre/time, `order`, `thresh`, `is_padding`) | — | Auger–Flandrin reassignment with the time-weighted and derivative windows; `order` repeats the reassignment | port as [`Reassign`](@ref) | fixture |
| `synsq` (synchrosqueezing of an existing CWT, `order`, `thresh`) | — | instantaneous frequency from the unwrapped phase difference, coefficient added to its band at the same time. audioFlux maps the octave and linear grids with a bin step of `(n-1)/n` of the true one, unwraps in float32 over the whole signal (cumulative error of up to a few 10⁻⁴ rad), ignores `thresh` below 1 and indexes the order > 1 remapping transposed; Audio911 maps to the nearest band, wraps each difference, honours `thresh` and remaps per sample | port as [`synsq`](@ref) (complex matrix) and [`Synsq`](@ref) (front end) | fixture (mel grid direct; octave grid through an emulation of audioFlux's mapping) |
| `wsst` (wavelet synchrosqueezed transform, all wavelets and scales) | — | CWT plus the derivative wavelet (`iω ψ̂`) for the instantaneous frequency, then synchrosqueezing; same band-mapping note as `synsq` | port as [`wsst`](@ref) and [`Wsst`](@ref) | fixture (mel direct, octave through the mapping emulation) |
| `CWD` (Choi–Williams), `WVD` (Wigner–Ville) | — | declared in `include/cwd_algorithm.h` and `wvd_algorithm.h` only: no C source, no Python wrapper in the snapshot | port from the literature as [`Wvd`](@ref) and [`Cwd`](@ref) (pseudo forms on the `Frames` grid, [`CohenDistribution`](@ref), signed values through [`get_distribution`](@ref)) and [`wvd`](@ref) (whole signal) | structural (time marginal, tone localisation, cross-term attenuation) |
| `EMD`, `EWT`, `HHT` | — | header-only declarations, no implementation in the snapshot | port from the literature: [`emd`](@ref) (Huang 1998 sifting, mirrored boundaries), [`ewt`](@ref)/[`Ewt`](@ref) (Gilles 2013), [`Hht`](@ref) (Hilbert spectrum of the IMFs on the `Frames` grid, [`get_imfs`](@ref)) | structural (completeness, tone separation, tight frame) |
| `Cepstrogram` (`cep_num`) | — (Audio911 has `pitch_cep` per frame) | real cepstrum `ifft(log|X|²)` of every STFT frame, liftered into an envelope (first `cep_num` quefrencies and their mirror) and details (the rest; audioFlux's details range includes one mirrored quefrency of the envelope, kept for parity) | port as [`Cepstrogram`](@ref) | fixture |
| `Temporal` (`energy`, `rms`, `zcr`, `ezr`) | [`Energy`](@ref), [`Rms`](@ref), [`Zcr`](@ref) | energy and rms of the windowed frame: `Energy(frames; windowed=true)`, `Rms(frames; windowed=true)`. audioFlux's zero-crossing rate counts strict sign changes of the windowed frame; Audio911's counts a zero as positive. `ezr = log10(1 + γ·energy) / (zcr·N + 1)` is new; `ezr` is not wrapped in Python (the fixture calls the C function) | extend existing: [`Ezr`](@ref), `Zcr(...; windowed, strict)` | fixture |
| `Spectrogram` class (`Linear`, `Mel`, `Bark`, `Erb`, `Chroma`, `Deep`, `DeepChroma`) | `Stft`, `MelSpec`, `BarkSpec`, `ErbSpec`, `Chroma` | `Linear`/`Mel`/`Bark`/`Erb` are BFT scale types (above). `Deep`/`DeepChroma` are an undocumented salience-filtered spectrogram (`SpectralDeepConfig`: `maxMin=13`, `minMax=2`, `ratio=10`, amplitude recovery and K-weighting flags) with no published definition | `Deep`, `DeepChroma`: out of scope (no definition to port beyond the C constants; nothing to validate against) | — |
| `is_continue` streaming mode of every transform | — | state carried across successive calls | out of scope (Audio911 processes a whole `Frames` object; streaming is listed as not implemented in the coverage table) | — |

## Frequency scales (`SpectralFilterBankScaleType`)

| audioFlux | Audio911 | definition | status | oracle |
|:----------|:---------|:-----------|:-------|:-------|
| `LINEAR` | `Stft` bins, [`LinSpec`](@ref) | one band per FFT bin | already covered | MATLAB |
| `LINSPACE` | — | `num` triangles with linearly spaced centres between `low_fre` and `high_fre` | port: scale function [`linspace`](@ref) | fixture |
| `MEL` | [`htk`](@ref) | `2595 log10(1 + f/700)`: identical to the HTK scale | already covered | MATLAB |
| `BARK` | [`bark`](@ref) | Traunmüller with the corrections below 2 and above 20.1 bark: identical | already covered | MATLAB |
| `ERB` | — (`ErbSpec` uses gammatone filters on the Glasberg–Moore ERB-rate) | ERB-rate `21.3654 log10(1 + 0.004368 f)` used to place triangular (or window-shaped) filters | port: scale function [`erb`](@ref) for `auditory_fbank` | fixture |
| `OCTAVE` | — (`Cwt` `voices` gives a geometric grid, `cqt_frequencies` the values) | `bin_per_octave` bins per octave anchored on 440 Hz, `low_fre` rounded to the nearest bin | port: scale function [`octave`](@ref) | fixture |
| `LOG` | — | `num` geometrically spaced centres between `low_fre` and `high_fre` | port: scale function [`logspace`](@ref) | fixture |
| `DEEP`, `CHROMA`, `LOG_CHROMA`, `DEEP_CHROMA` (`SpectralFilterBankType` only) | `Chroma` | chroma folding of the linear and octave scales (see features); `DEEP*` out of scope as above | see `chroma` rows | — |

## Filter styles and normalisation

| audioFlux | Audio911 | definition | status | oracle |
|:----------|:---------|:-----------|:-------|:-------|
| `SLANEY` style | [`auditory_fbank`](@ref) (`domain=:linear`) | triangles between the neighbouring centres on the frequency grid: identical | already covered | MATLAB |
| `ETSI` style | [`etsi_fbank`](@ref) | audioFlux's "ETSI" is a triangle on the nearest grid bins without the `+1` offsets of ETSI ES 201 108 that `etsi_fbank` implements: different | port: `style=etsi` | fixture |
| `GAMMATONE` style | [`gammatone_fbank`](@ref) | 4th-order gammatone magnitude response: identical construction | already covered | MATLAB |
| `POINT`, `RECT`, `HANN`, `HAMM`, `BLACKMAN`, `BOHMAN`, `KAISER`, `GAUSS` styles | — | the band is a window of the given shape spanning the two neighbouring centres (`POINT` keeps only the centre bin) | port: `style=` keyword of `auditory_fbank` (`triangular`, `etsi`, `point`, `rect`, `hanning`, `hamming`, `blackman`, `bohman`, `kaiser`, `gauss`) | fixture |
| normalisation `NONE`, `AREA`, `BAND_WIDTH` | [`none_norm`](@ref), [`area`](@ref), [`bandwidth`](@ref) | same three rules (`area` divides by the filter sum, `bandwidth` by half the distance between neighbouring centres) | already covered | MATLAB |
| window types `RECT … TUKEY` (14 windows) | DSP windows plus [`povey`](@ref) | audioFlux adds `flattop`, `gauss`, `kaiser`, `blackman_harris`, `blackman_nuttall`, `bohman`, `tukey`; DSP.jl provides all but `bohman` | extend existing: `bohman` window, the DSP ones reexported | structural |

## Features: `spectral`

All on any spectrogram (audioFlux `Spectral`, `spectrogramObj_*`, `bftObj`).
"Definition" quotes `src/flux_spectral.c`.

| audioFlux | Audio911 | definition | status | oracle |
|:----------|:---------|:-----------|:-------|:-------|
| `flatness` | [`SpectralFlatness`](@ref) | `exp(mean(log(S + 2e-16))) / mean(S)`: identical up to the epsilon | already covered | MATLAB |
| `flux(step, p, is_positive, is_exp, tp)` | [`SpectralFlux`](@ref) (`p`-norm with root) | audioFlux: `Σ |S_t − S_{t−step}|^p`, optional half-wave rectification, optional mean instead of sum, root only with `is_exp`; its default (`p=2`, no root) is the squared MATLAB flux | extend existing: keywords `step`, `positive`, `root`, `mean` (defaults keep the MATLAB definition) | fixture |
| `rolloff(threshold)` | [`SpectralRolloff`](@ref) | first bin where the cumulative sum reaches `threshold · Σ S`: identical | already covered | MATLAB |
| `centroid` | [`SpectralCentroid`](@ref) | identical | already covered | MATLAB |
| `spread` | [`SpectralSpread`](@ref) | identical | already covered | MATLAB |
| `skewness`, `kurtosis` | [`SpectralSkewness`](@ref), [`SpectralKurtosis`](@ref) | identical (standardised third and fourth moments) | already covered | MATLAB |
| `entropy(is_norm)` | [`SpectralEntropy`](@ref) | Audio911 always normalises by `log2(nbins)` (MATLAB); audioFlux normalises only with `is_norm=True` | extend existing: `normalize` keyword | fixture |
| `crest` | [`SpectralCrest`](@ref) | identical | already covered | MATLAB |
| `slope` | [`SpectralSlope`](@ref) | identical | already covered | MATLAB |
| `decrease` | [`SpectralDecrease`](@ref) | identical | already covered | MATLAB |
| `band_width(p)` | [`SpectralBandwidth`](@ref) | audioFlux: `(Σ S (f − c)^p)^(1/p)` without normalising `S` to unit sum (librosa and Audio911 normalise), and with the signed deviation raised to `p` (Audio911 uses `|f − c|^p`, which differs for `p ≠ 2`) | extend existing: `normalize` keyword | fixture (`p = 2`) |
| `rms` | [`Rms`](@ref)`(spec)` | audioFlux: `sqrt(2 Σ' S² / nbins²)` with half weight on DC (and on the last bin when their number is even), `S` the magnitude; Audio911's `Rms` divides by `nfft²` (librosa) | extend existing: [`SpectralRms`](@ref) with audioFlux's normalisation | fixture |
| `energy(is_log, gamma)` | — | `mean(S²)` over the bins (`S` if power), optionally `log(1 + γ S²)` | port: [`SpectralEnergy`](@ref) | fixture |
| `hfc` | — | high-frequency content `Σ k · S_k` (bin index weighted) | port: [`SpectralHfc`](@ref) | fixture |
| `sd(step, is_positive)` | — | spectral difference `Σ |S_t − S_{t−step}|` | port: [`SpectralSd`](@ref) | fixture |
| `sf(step, is_positive)` | — | `Σ (S_t − S_{t−step})²` | port: [`SpectralSf`](@ref) | fixture |
| `mkl(tp)` | — | modified Kullback–Leibler `Σ log(1 + S_t / (S_{t−1} + ε))`, sum or mean | port: [`SpectralMkl`](@ref) | fixture |
| `pd`, `wpd`, `nwpd` | — | phase deviation `mean |φ_t − 2φ_{t−1} + φ_{t−2}|`, weighted by the magnitude, normalised by the mean magnitude | port: [`SpectralPd`](@ref), [`SpectralWpd`](@ref), [`SpectralNwpd`](@ref) (need the phase, see the complex accessor) | fixture |
| `cd`, `rcd` | — | complex deviation `Σ |S_t e^{iφ_t} − S_{t−1} e^{i(2φ_{t−1} − φ_{t−2})}|`, rectified variant counts only rising bins | port: [`SpectralCd`](@ref), [`SpectralRcd`](@ref) | fixture |
| `broadband(threshold)` | — | number of bins whose level rises by more than `threshold` dB from the previous frame | port: [`SpectralBroadband`](@ref) | fixture |
| `novelty(step, threshold, method_type, data_type)` | — | per-bin novelty `sub`/`entropy`/`kl`/`is` against frame `t−step`, summed (`value`) or counted (`number`) where above `threshold` | port: [`SpectralNovelty`](@ref) | fixture |
| `eef(is_norm)` | — | energy–entropy feature `sqrt(1 + |energy · entropy|)` | port: [`SpectralEef`](@ref) | fixture |
| `eer(is_norm, gamma)` | — | `sqrt(1 + |log(1 + γ energy) / entropy|)` | port: [`SpectralEer`](@ref) | fixture |
| `max`, `mean`, `var` | — | per-frame maximum (value and its frequency), mean and sample variance of the values; the frequency outputs of `mean` and `var` are the mean and variance of the band frequencies, the same for every frame | port: [`SpectralMax`](@ref), [`SpectralPeak`](@ref) (frequency of the maximum), [`SpectralMean`](@ref), [`SpectralVar`](@ref); the constant frequency outputs are not ported | fixture |
| `set_edge`, `set_edge_arr` (bin subsets) | `freqrange` on [`LinSpec`](@ref) | descriptors on a bin range are descriptors of a `LinSpec` | already covered | — |

## Features: cepstra, deconvolution, chroma

| audioFlux | Audio911 | definition | status | oracle |
|:----------|:---------|:-----------|:-------|:-------|
| `xxcc(cc_num, rectify_type)` on any spectrogram | [`Mfcc`](@ref) on any spectrogram | `log10` (audioFlux calls it "matlab canonic", floored at `1e-8`) or cubic-root rectification followed by an orthonormal DCT-II: identical to `Mfcc(spec; rect=mlog, floor=1e-8)` and `rect=cubic_root` | already covered | fixture (`Mfcc` against `xxcc`) |
| `xxcc_standard(cc_num, delta_window_length, energy_type, rectify_type)` | [`Mfcc`](@ref) + [`Delta`](@ref) | coefficients with the natural-log energy replacing C0 or, for audioFlux's `APPEND`, placed *before* the coefficients, plus delta and delta-delta along the coefficient axis (`Delta(source=:transposed)`). The 0.1.9 Python wrapper writes past its buffers with `APPEND`; the fixture calls the C function | extend existing: [`xxcc_standard`](@ref), `energy_mode=:prepend` on [`Mfcc`](@ref) | fixture |
| `mfcc`, `bfcc`, `gtcc`, `cqcc` (`core.py`) | `Mfcc(MelSpec)`, `Mfcc(BarkSpec)`, `Gtcc(ErbSpec)`, `Mfcc(Cqt)` | the four are `xxcc` on the mel, bark, ERB and CQT spectrograms | already covered (`cqcc` once `Cqt` exists) | fixture |
| `cqhc` (CQT harmonic coefficients) | — | undocumented (`cqtObj_cqhc`, harmonic folding of the CQT) | out of scope (no definition beyond the C loop; not exposed in Python) | — |
| `deconv` on any spectrogram | — | per frame: `|FFT(S)|` of the magnitude spectrum; timbre = `real(ifft(|FFT(S)|))`, pitch = `real(ifft(FFT(S) / |FFT(S)|))` | port: [`Deconv`](@ref) | fixture |
| `chroma_linear` (`chroma_stftFilterBank`) | [`Chroma`](@ref), [`chroma_fbank`](@ref) | the librosa chroma filterbank (Gaussian bins, octave weighting, L2 columns, C first) with per-frame max normalisation: identical | already covered | fixture (`Chroma(Stft)` against `chroma_linear`, 1e-6) |
| `chroma_octave` | — | chroma of an octave-scale BFT | already covered once the `octave` scale exists (`Chroma(MelSpec(stft; scale=octave))` folds the bands) | structural |
| `chroma_cqt` (`chroma_cqtFilterBank`, `ChromaDataNormalType` none/max/min/p1/p2) | — | fold the CQT bins of each octave into `chroma_num` classes (rectangular filters, `bin_per_octave/chroma_num` bins each), per-frame normalisation | port: `Chroma(cqt::Cqt)` with the folding filterbank; norm `nothing`/`Inf`/`1`/`2` (`min` is not offered) | fixture |
| `FeatureExtractor` (many transforms in one call) | pipeline composition | convenience wrapper | already covered (compose stages) | — |

## MIR

| audioFlux | Audio911 | definition | status | oracle |
|:----------|:---------|:-----------|:-------|:-------|
| `PitchYIN(low_fre, high_fre, auto_length, thresh=0.1)` | [`pitch_yin`](@ref) | de Cheveigné's cumulative-mean-normalised difference with parabolic interpolation: same definition; audioFlux returns 0 for frames without a dip below the threshold, Audio911 the global minimum | already covered | fixture (agreement within half a semitone on a harmonic sweep ≥ 95%) |
| `PitchNCF` | [`pitch_ncf`](@ref) | autocorrelation over the lag range: audioFlux (MATLAB's NCF) normalises by the lag-0 energy and takes the integer lag, Audio911 by the energy of the overlapping segments with parabolic interpolation | already covered | fixture (sweep agreement ≥ 95%) |
| `PitchCEP` | [`pitch_cep`](@ref) | real-cepstrum peak: same definition; audioFlux reports `sr/(q+1)` for the peak quefrency `q`, Audio911 `sr/q` with interpolation | already covered | fixture (sweep agreement ≥ 95%) |
| `PitchPEF(cut_fre, alpha, beta, gamma)` | — | pitch estimation filter (Gonzalez & Brookes 2011; MATLAB `pitch` `"PEF"`): log-frequency power spectrum, comb-like filter with `α`, `β`, `γ`, correlation peak | port: [`pitch_pef`](@ref) | fixture (identical frame by frame) |
| `PitchHPS(harmonic_count)` | — | harmonic product spectrum on a ~1 Hz grid, `harmonic_count` multiples; audioFlux searches the Hz bounds as bin indices and reports `(bin + 1)·sr/N`, Audio911 searches the bins inside the range and reports `bin·sr/N` | port: [`pitch_hps`](@ref) | fixture (identical bins, one-bin offset) |
| `PitchLHS(harmonic_count)` | — | log-harmonic summation (MATLAB `"LHS"`); same grid notes as HPS | port: [`pitch_lhs`](@ref) | fixture (identical bins, one-bin offset) |
| `PitchSTFT` | — | spectral-peak pitch: STFT peaks in dB with window-specific frequency correction and a harmonic vote (`trist`, ~1000 lines of undocumented heuristics). In 0.1.9 its constructor swaps the bin bounds (the range is ignored) and it misses the fundamental of a clean harmonic sweep | port (simplified): [`pitch_stft`](@ref), quadratic peak interpolation and a harmonic sieve; `trist` is not reproduced | structural (ground-truth sweep) |
| `PitchFFP` | — | "flux fast pitch", a 3000-line undocumented tracker with correlation, cut, flag, light and temporal side outputs | out of scope (no published definition; only the C source, which is a heuristic state machine) | — |
| `Onset(novelty_type, filter_order)` + `NoveltyParam` | [`OnsetStrength`](@ref), [`onset_detect`](@ref), [`peak_pick`](@ref) | audioFlux's envelope is the chosen spectral novelty (flux/hfc/sd/sf/mkl/pd/wpd/nwpd/cd/rcd/broadband) after an optional running maximum over `filter_order` bins, min–max normalised, then librosa's `peak_pick` with the same defaults. Audio911's envelope is librosa's dB flux | extend existing: [`Novelty`](@ref) envelope (any of the novelty descriptors, audioFlux normalisation) and `onset_detect` on any descriptor | fixture |
| `HPSS(h_order, p_order)` (median filtering) | [`Hpss`](@ref) | same median filters (harmonic along time, percussive along frequency) with hard masks; audioFlux returns the two time-domain signals through an ISTFT, Audio911 the masked spectrograms | extend existing: `Hpss(...; power=Inf)` is the audioFlux mask; [`get_harmonic_signal`](@ref)/[`get_percussive_signal`](@ref) through `istft` | fixture |
| HPSS by NMF (README) | — | listed in the README; the snapshot has no NMF-based HPSS (only `classic/nmf.c`) | port structurally: `Hpss(spec; method=nmf, k=...)` classifying NMF components by their spectral versus temporal continuity | structural |
| `Harmonic.harmonic_count(low_fre, high_fre)` | — | STFT peaks in dB filtered by height, neighbourhood and level, counted between `low` and `high` | port: [`harmonic_count`](@ref) (the three peak filters) | structural (the filter constants are undocumented; the count is checked on synthetic harmonic tones) |
| `HarmonicRatio(low_fre)` | [`HarmonicRatio`](@ref) | audioFlux: normalised autocorrelation of the windowed frame zero-padded to twice its length, maximised from the first zero crossing of the autocorrelation (kept from the previous frame when there is none) up to `sr/low_fre`, lag energy one sample short of the overlap, quadratic interpolation; Audio911's (MATLAB) maximises over the lags of a frequency range | extend existing: `HarmonicRatio(frames; method=:audioflux, fmin)` | fixture |
| `PitchShift(n_semitone)` | — | phase vocoder time stretch by `2^(n/12)` then resampling | port: [`pitch_shift`](@ref) | fixture |
| `TimeStretch(rate)` | — | phase vocoder (`dsp/phase_vocoder.c`) and weighted overlap-add ISTFT | port: [`time_stretch`](@ref), [`phase_vocoder`](@ref), [`istft`](@ref) | fixture |
| `TuneTrack` (C only, no Python wrapper) | — | instrument tuner: pitch tracking with note, cents, dB and stability outputs; 1700 lines of state machine | port structurally simplified: [`tune_track`](@ref) (YIN pitch to nearest note, cents deviation, median-smoothed) | structural |

## DSP utilities

| audioFlux | Audio911 | definition | status | oracle |
|:----------|:---------|:-----------|:-------|:-------|
| `CZT(low_w, high_w)` | — | chirp-Z transform of a `2^k` block over the normalised frequency band `[low_w, high_w]` (Bluestein) | port: [`czt`](@ref) | fixture |
| `hilbert` (C only) | — | analytic signal through the FFT | port: [`hilbert`](@ref) | structural (analytic-signal identities) |
| `Xcorr(normal_type none/coeff)` | [`autocorrelate`](@ref) (auto only) | full cross-correlation `2N−1` lags, optional `coeff` normalisation | port: [`xcorr`](@ref) | fixture |
| `conv(mode full/same/valid, method auto/direct/fft)` (C only) | — | linear convolution | port: [`convolve`](@ref) | structural |
| `Resample(quality best/mid/fast, is_scale)`, `WindowResample(zero_num, nbit, win_type, value, roll_off)` | [`resample`](@ref) (polyphase FIR) | audioFlux's `POLYPHASE` is a polyphase FIR (matches), `BANDLIMITED` is the CCRMA windowed-sinc with a lookup table | extend existing: `resample(...; method=:sinc, zeros, roll_off, window)` | fixture |
| `phase_vocoder` | — | standard phase advance by `rate` on the complex STFT | port (with `time_stretch`) | fixture |
| `auditory_weight_a/b/c/d` | [`A_weighting`](@ref), [`C_weighting`](@ref) | A and C: same IEC 61672 curves (audioFlux uses 12200 Hz for A instead of 12194, floored at −80 dB). B and D are new | extend existing: [`B_weighting`](@ref), [`D_weighting`](@ref) | fixture |
| `dct`, `dft`, `fft` (internal) | FFTW, [`dct_ortho`](@ref) | internal helpers | already covered | — |
| FIR/IIR design, `freqz` (`src/dsp/filterDesign_*`, internal) | DSP.jl | internal helpers not exposed in Python | out of scope (DSP.jl provides filter design) | — |

## Classic

| audioFlux | Audio911 | definition | status | oracle |
|:----------|:---------|:-----------|:-------|:-------|
| `nmf(k, max_iter, tp kl/is/euc, thresh, norm)` | — | multiplicative updates for the KL, Itakura–Saito and Euclidean divergences, column normalisation max/sum/L2 | port: [`nmf`](@ref) | fixture |
| `hmm` (C only: init, predict, decode, train, generate) | — | discrete HMM: forward likelihood, Viterbi decoding, Baum–Welch training, sampling | port: [`Hmm`](@ref) with `predict`, `decode`, `fit!`, `generate` | structural |
| `viterbi` (C only) | — | Viterbi path with log or linear probabilities | port: [`viterbi`](@ref) | structural |
| `trist` (internal) | — | helper of the STFT pitch method | folded into `pitch_stft` | — |

## Utilities (`audioflux.utils`)

| audioFlux | Audio911 | definition | status | oracle |
|:----------|:---------|:-----------|:-------|:-------|
| `power_to_db(min_db)` | [`power_to_db`](@ref) | `10 log10(S / max)` floored at `min_db`: `power_to_db(S; ref=maximum, top_db=-min_db)` | already covered | structural |
| `power_to_abs_db`, `mag_to_abs_db(fft_length, is_norm, min_db)` | — | absolute dB relative to `fft_length` (and to the window sum when `is_norm`) | port: [`power_to_abs_db`](@ref), `mag_to_abs_db` | fixture |
| `log_compress(gamma)`, `log10_compress(gamma)` | — | `log(1 + γ S)`, `log10(1 + γ S)` | port: [`log_compress`](@ref) | fixture |
| `temproal_db(base)` | — | per-frame dB statistics (max, average, percentage above `base`) | port: [`temporal_db`](@ref) | fixture |
| `delta(order)` | [`Delta`](@ref) | regression filter along the last axis: `Delta(source=:transposed)` | already covered | — |
| `get_phase` | [`get_phase`](@ref) | `atan2(imag, real)` | port (with the complex accessor) | — |
| `note_to_midi`, `midi_to_hz`, `note_to_hz`, `midi_to_note`, `hz_to_midi`, `hz_to_note` | same names | identical | already covered | — |
| `min_max_scale`, `stand_scale`, `max_abs_scale`, `robust_scale`, `center_scale`, `mean_scale`, `arctan_scale` | — | per-row feature scaling | port: same names | structural |
| `synth_f0(times, frequencies, samplate, amplitudes)` | [`tone`](@ref) (constant frequency) | additive synthesis of a pitch curve | port: [`synth_f0`](@ref) | structural |
| `queue_fre2`, `queue_fre3` | — | "queue frequency" ratios of two or three frequencies, implemented in the 7700-line `mir/_queue.c` without documentation | out of scope (no definition) | — |
| `read`, `write`, `convert_mono`, `resample`, `chirp` (`audio.py`) | [`load`](@ref), [`to_mono`](@ref), [`resample`](@ref), [`chirp`](@ref) | audioFlux delegates to soundfile/scipy; `write` has no counterpart | `write`: out of scope (Audio911 is an analysis library; the loader is read-only) | — |
| `sample_path`, `check_audio`, `ascontiguous_*` | — | Python conveniences | out of scope (n/a in Julia) | — |
| `display` (`fill_spec`, `fill_wave`, `fill_plot`, `Plot`) | Plots recipes | plotting | already covered (every new type gets a recipe) | — |

## Summary of the plan

Ports, in the order they are delivered: octave/log/linspace/ERB scales and
window filter styles; `Cqt` (with VQT); `Pwt`; `St`/`fst`; `Nsgt`; the
complex-spectrum accessor and `istft`; `Reassign`; `Synsq` and `Wsst`; the
extra wavelets and grids of `Cwt`; `dwt`/`wpt`/`swt` with the wavelet
tables; `Wvd`, `Cwd`, `emd`, `ewt`, `Hht`; the spectral descriptors;
`Deconv`, `Cepstrogram`, `Ezr`, `xxcc_standard`; `Novelty`, the pitch
methods, `harmonic_count`, the `HarmonicRatio` variant, HPSS signals and NMF
HPSS, `nmf`, `Hmm`, `viterbi`, `tune_track`; `phase_vocoder`,
`time_stretch`, `pitch_shift`; `czt`, `hilbert`, `xcorr`, `convolve`, the
sinc resampler, B/D weightings and the scaling utilities.

Out of scope, with the reason given in the rows: the `Deep` spectrograms,
`cqhc`, `PitchFFP`, `queue_fre*`, `is_continue` streaming, audio writing and
the internal filter-design helpers.
