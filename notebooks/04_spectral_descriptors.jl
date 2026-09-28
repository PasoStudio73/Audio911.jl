### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ e09a9adb-5474-41cf-8737-c5b375eb76fc
begin
	import Pkg
	# the notebook environment next to this file pins Audio911 (from this repository), PlutoUI and Plots
	Pkg.activate(@__DIR__)
	Pkg.instantiate()
	using Audio911, PlutoUI, Plots
	include(joinpath(@__DIR__, "common.jl"))
	gr(); default(size=(720, 300), legend=:topright, titlefontsize=10, guidefontsize=9)
end

# ╔═╡ a5bf0961-daa0-4af0-8782-7d07089eadb1
md"""
# Audio911 · 4 — Spectral descriptors

A **spectral descriptor** summarises every frame of a spectrogram with a single number: where the energy sits, how peaky or noisy the spectrum is, how fast it changes.
All of them are `AbstractSpectral` objects: `get_data(d)` is one value per frame and `get_times(d)` is the parent's time axis, so every descriptor lines up with the spectrogram it came from.

Descriptors are written against the common spectrogram interface, so the **same descriptor runs on an `Stft`, a `LinSpec`, a `MelSpec` or a `Cwt`**.
The source you pick changes the answer: a mel spectrogram weights low frequencies more densely than a linear one, so its centroid is lower; a scalogram has a geometric frequency grid.

This notebook covers the 31 descriptors of `src/fft/spectral.jl` (MATLAB `audioFeatureExtractor` definitions) and `src/fft/spectral_af.jl` (audioFlux definitions), plus `SpectralContrast` and `PolyFeatures`.
"""

# ╔═╡ a642ed41-4014-4927-bc50-232dea7859a7
TableOfContents()

# ╔═╡ cbe30f27-1a3c-493f-acd8-5d8d63dc584d
md"""
## 1. Signal and source spectrogram

Signal: $(@bind signal_name Select(SIGNALS))
winsize: $(@bind winsize Select([256, 512, 1024]; default=512))
hop: $(@bind winstep Select([64, 128, 256]; default=256))
"""

# ╔═╡ f803212b-1860-4c3b-ab70-5c66687fdef7
begin
	audio  = demo_signal(signal_name)
	sr     = get_sr(audio)
	frames = Frames(audio; winsize, winstep=min(winstep, winsize))
	# keep_complex=true stores the complex STFT so the phase-based descriptors of section 6 do not recompute it
	stft   = Stft(frames; keep_complex=true)
end;

# ╔═╡ e74f33fa-f009-4dde-9bdf-f20e02f34095
audio_player(audio)

# ╔═╡ 4e9d4e11-ebd7-4692-8271-5136a92a861a
md"""
The descriptors read whatever spectrogram they are given.

| control | effect on the descriptors |
|:--|:--|
| source | `Stft`: every FFT bin, uniformly `sr/nfft` apart. `LinSpec`: the same bins, restricted to a frequency range and one-sided scaled (the MATLAB `audioFeatureExtractor` input). `MelSpec`: a few bands on the mel scale, dense at low frequency. `Cwt`: a scalogram with `voices` scales per octave. |
| spectrum | `power` (`|X|²`) or `magnitude` (`|X|`) STFT. Squaring exaggerates the loudest bins, so a power spectrum gives lower entropy and flatness and a centroid pulled toward the strongest partial. |
| `freqrange` (LinSpec) | only bins inside the range count. Cutting the low end removes rumble from centroid and slope; cutting the high end lowers centroid and rolloff. |
| `nbands` (MelSpec) | number of mel bands: fewer bands smooth the spectrum and make shape descriptors less noisy. |
| `voices` (Cwt) | scales per octave of the wavelet grid. |
"""

# ╔═╡ c46eddc8-74f6-4403-8bfa-740a8f552c2c
md"""
source: $(@bind source_kind Select(["Stft", "LinSpec", "MelSpec", "Cwt"]; default="LinSpec"))
spectrum: $(@bind spectrum_kind Select([power => "power", magnitude => "magnitude"]))

LinSpec range: $(@bind lin_lo Slider(0:50:1500; default=0, show_value=true)) – $(@bind lin_hi Slider(2000:250:8000; default=8000, show_value=true)) Hz
MelSpec bands: $(@bind mel_bands Slider(8:4:80; default=40, show_value=true))
Cwt voices: $(@bind cwt_voices Slider(4:2:24; default=12, show_value=true))
"""

# ╔═╡ 27e12e5e-9508-4fec-91f7-f80a167059b8
begin
	base = Stft(frames; spectrum=spectrum_kind, keep_complex=true)
	src = source_kind == "Stft"    ? base :
	      source_kind == "LinSpec" ? LinSpec(base; freqrange=(lin_lo, lin_hi)) :
	      source_kind == "MelSpec" ? MelSpec(base; nbands=mel_bands) :
	      Cwt(frames; voices=cwt_voices, freqrange=(60, sr ÷ 2), spectrum=spectrum_kind)
end

# ╔═╡ a0e21baa-58ff-48b5-bd7f-1cf4e7af7b9a
plot(src; title="source: $source_kind ($(nameof(spectrum_kind)))")

# ╔═╡ 7d74db85-4a98-464b-8180-7b0e7ef39f9f
md"""
## 2. Where the energy sits: spectral moments

Treat each frame's spectrum `S(f)` as a distribution over frequency.

| descriptor | formula | meaning |
|:--|:--|:--|
| `SpectralCentroid` | `μ = Σ f·S / Σ S` | centre of mass in Hz, the "brightness". |
| `SpectralSpread` | `σ = √(Σ (f-μ)²·S / Σ S)` | standard deviation around the centroid in Hz: narrow for a tone, wide for noise. |
| `SpectralBandwidth(; p, normalize)` | `(Σ p̃ |f-μ|^p)^(1/p)` | librosa's generalisation of spread. `p=2` with `normalize=true` equals `SpectralSpread`; larger `p` weights far-away energy more; `normalize=false` weights by `S` itself (audioFlux `band_width`), so it scales with level. |
| `SpectralRolloff(; threshold)` | smallest `f` with `Σ_{≤f} S ≥ threshold·Σ S` | frequency below which `threshold` of the energy lies. `0.95` tracks the effective bandwidth; `0.5` is a median frequency. |
| `SpectralSkewness` | `Σ (f-μ)³S / (σ³ Σ S)` | asymmetry: positive when energy is concentrated low with a long high tail. |
| `SpectralKurtosis` | `Σ (f-μ)⁴S / (σ⁴ Σ S)` | peakedness: large for a single dominant partial, about 1.8 for a flat spectrum. |

rolloff threshold: $(@bind rolloff_th Slider(0.05:0.05:0.99; default=0.95, show_value=true))
bandwidth `p`: $(@bind bw_p Slider(0.5:0.5:4; default=2, show_value=true))
bandwidth normalize $(@bind bw_norm CheckBox(default=true))
"""

# ╔═╡ 85562057-4b9c-481e-b8af-2bc5f56b060f
begin
	centroid  = SpectralCentroid(src)
	spread    = SpectralSpread(src)
	bandwidth = SpectralBandwidth(src; p=bw_p, normalize=bw_norm)
	rolloff   = SpectralRolloff(src; threshold=rolloff_th)
	skewness  = SpectralSkewness(src)
	kurtosis  = SpectralKurtosis(src)
end;

# ╔═╡ 9cbd5df4-735c-42c5-af7d-58a1e6fa228d
let t = get_times(centroid), c = get_data(centroid), s = get_data(spread)
	p = plot(src; title="centroid ± spread and rolloff on the $source_kind", colorbar=false)
	plot!(p, t, c; c=:white, lw=2, label="centroid")
	plot!(p, t, [c .- s c .+ s]; c=:white, ls=:dash, lw=1, label=["centroid ± spread" ""])
	plot!(p, t, get_data(rolloff); c=:cyan, lw=2, label="rolloff $(rolloff_th)")
	bw_norm && plot!(p, t, c .+ get_data(bandwidth); c=:orange, ls=:dot, lw=1.5, label="centroid + bandwidth (p=$bw_p)")
	plot!(p; size=(720, 360), legend=:topright)
end

# ╔═╡ 32d096c9-dd08-4a65-a874-452e661ecdf3
plot(plot(skewness; c=1), plot(kurtosis; c=2); layout=(2, 1), size=(720, 380))

# ╔═╡ ef5a7597-1a38-481c-ae38-e38adfe1b9b5
md"""
On the **chirp** the centroid follows the sweep exactly and the spread stays small; on **speech** the centroid jumps up on fricatives (*s*, *f*) and the kurtosis spikes on voiced vowels dominated by a few harmonics.
Switch the source to `MelSpec`: the centroid drops, because the mel bands put most of their weight below 2 kHz.
"""

# ╔═╡ 0df65504-5f59-4adb-a97c-057a49bc8df7
md"""
## 3. Spectral shape: tonal or noisy?

| descriptor | formula | meaning |
|:--|:--|:--|
| `SpectralFlatness` | geometric mean / arithmetic mean of `S` | 1 for white noise, near 0 for a tone (Wiener entropy). |
| `SpectralEntropy(; normalize)` | `-Σ p log₂ p`, `p = S/ΣS` | how spread the energy is over the bins. `normalize=true` divides by `log₂(nbins)` so the result is in [0, 1] regardless of the number of bins (MATLAB); `false` leaves bits (audioFlux `is_norm=False`). |
| `SpectralCrest` | `max(S) / mean(S)` | how much the strongest bin stands out; high for tones. |
| `SpectralDecrease` | `Σ_{k≥2} (S_k - S_1)/(k-1) / Σ_{k≥2} S_k` | average slope from the first bin, weighted toward low bins (perceptually motivated). |
| `SpectralSlope` | least-squares slope of `S` against `f` | overall tilt; negative for most natural sounds. |

entropy normalize $(@bind ent_norm CheckBox(default=true))
"""

# ╔═╡ d5234530-1050-4f78-91af-dc7bd4f90c8e
let ds = [SpectralFlatness(src), SpectralEntropy(src; normalize=ent_norm), SpectralCrest(src),
          SpectralDecrease(src), SpectralSlope(src)]
	plot([plot(d; c=i, legend=false, yguide="", title=string(nameof(typeof(d)))) for (i, d) in enumerate(ds)]...;
	     layout=(5, 1), size=(720, 700), xguide="")
end

# ╔═╡ cdfb7701-b62e-42e8-97a9-4612fc0ec2f7
md"""
Pick **Two close tones and clicks**: flatness and entropy jump at every click (broadband), while crest collapses there and is high elsewhere.
"""

# ╔═╡ 5378e434-b13c-4eaa-8ea7-1bbe1ec6f41b
md"""
## 4. Change between frames: flux and novelty

These compare frame `t` with frame `t - step`; the first `step` frames are 0. They are the raw material of onset detection (notebook 7).

| descriptor | formula | keywords |
|:--|:--|:--|
| `SpectralFlux(; p, step, positive, root, mean)` | `(Σ |S_t - S_{t-step}|^p)^(1/p)` | `p`: norm (1 = sum of differences, 2 = Euclidean, large = dominated by the biggest change). `step`: lag; larger lags react to slower changes and smooth the curve. `positive`: keep only increases (half-wave rectification), so offsets are ignored. `root=false`: skip the `1/p` power (audioFlux default). `mean`: divide by the number of bins. |
| `SpectralSd(; step, positive)` | `Σ |S_t - S_{t-step}|` | audioFlux spectral difference. |
| `SpectralSf(; step, positive)` | `Σ (S_t - S_{t-step})²` | squared difference; emphasises large changes. |
| `SpectralMkl(; mean)` | `Σ log(1 + S_t / S_{t-1})` | modified Kullback–Leibler; sensitive to relative changes, so quiet onsets count. |
| `SpectralBroadband(; threshold)` | number of bins rising by more than `threshold` dB | counts rather than sums; a higher threshold keeps only sharp attacks. |
| `SpectralNovelty(; step, threshold, method, data)` | per-bin term summed (`data=:value`) or counted (`:number`) when above `threshold` | `method`: `:sub` difference, `:entropy` log ratio, `:kl` `S log(S/S')`, `:is` Itakura–Saito. |
"""

# ╔═╡ 90981d4a-862d-41ca-a572-c0700beb3f02
md"""
flux p: $(@bind flux_p Slider(0.5:0.5:4; default=2, show_value=true))
step: $(@bind flux_step Slider(1:8; default=1, show_value=true))
positive $(@bind flux_pos CheckBox(default=false))
root $(@bind flux_root CheckBox(default=true))
mean $(@bind flux_mean CheckBox(default=false))

MKL mean $(@bind mkl_mean CheckBox(default=false))
broadband threshold (dB): $(@bind bb_th Slider(0:1:30; default=0, show_value=true))

novelty method: $(@bind nov_method Select([:sub, :entropy, :kl, :is]))
data: $(@bind nov_data Select([:value, :number]))
threshold: $(@bind nov_th Slider(0:0.5:10; default=0, show_value=true))
"""

# ╔═╡ 55d6d7b9-ebc5-4559-b06b-a593063acebc
let ds = [SpectralFlux(src; p=flux_p, step=flux_step, positive=flux_pos, root=flux_root, mean=flux_mean),
          SpectralSd(src; step=flux_step, positive=flux_pos),
          SpectralSf(src; step=flux_step, positive=flux_pos),
          SpectralMkl(src; mean=mkl_mean),
          SpectralBroadband(src; threshold=bb_th),
          SpectralNovelty(src; step=flux_step, threshold=nov_th, method=nov_method, data=nov_data)]
	plot([plot(d; c=i, legend=false, yguide="", title=string(nameof(typeof(d)))) for (i, d) in enumerate(ds)]...;
	     layout=(6, 1), size=(720, 820), xguide="")
end

# ╔═╡ 901c6812-76c4-46d0-a027-e5f306507e6a
md"""
`step` and `positive` apply to flux, `Sd`, `Sf` and novelty together.
With `positive` on, the release of each click disappears and only the attacks remain; that is why onset detectors half-wave rectify.
"""

# ╔═╡ 0b5591d4-c5ce-4a9d-88d7-8c1953a42a40
md"""
## 5. Level and simple statistics

| descriptor | formula | keywords |
|:--|:--|:--|
| `SpectralEnergy(; log, gamma)` | `mean(|X|²)` or `mean(log(1 + γ|X|²))` | `log=true` compresses the dynamic range; a larger `gamma` makes the log curve bite earlier. |
| `SpectralRms` | `√(2 Σ w_k |X_k|² / n²)` | RMS level from the magnitude spectrum (DC and Nyquist at half weight). |
| `SpectralHfc` | `Σ k·S_k` | high-frequency content: energy weighted by bin index; spikes on percussive attacks. |
| `SpectralMax` / `SpectralPeak` | `max S` / its frequency | loudest bin value / its frequency in Hz. |
| `SpectralMean` / `SpectralVar` | mean / sample variance over bins | first two statistics of the bin values. |
| `SpectralEef(; normalize)` | `√(1 + |E·H|)` | energy × entropy: high for loud, noisy frames. |
| `SpectralEer(; normalize, gamma)` | `√(1 + |log(1 + γE) / H|)` | energy / entropy: high for loud, tonal frames. |

Energy and RMS always use the power (`|X|²`) and magnitude views of the source, whatever `spectrum` you chose; the others read the source values as given.

energy log $(@bind en_log CheckBox(default=false))
energy γ: $(@bind en_gamma Slider(0.5:0.5:50; default=10, show_value=true))
EEF/EER entropy normalize $(@bind ee_norm CheckBox(default=false))
EER γ: $(@bind eer_gamma Slider(0.1:0.1:10; default=1, show_value=true))
"""

# ╔═╡ 80eccd0f-70b4-4b99-8cd1-d139faa6903a
let ds = [SpectralEnergy(src; log=en_log, gamma=en_gamma), SpectralRms(src), SpectralHfc(src),
          SpectralMax(src), SpectralMean(src), SpectralVar(src),
          SpectralEef(src; normalize=ee_norm), SpectralEer(src; normalize=ee_norm, gamma=eer_gamma)]
	plot([plot(d; c=i, legend=false, yguide="", title=string(nameof(typeof(d)))) for (i, d) in enumerate(ds)]...;
	     layout=(4, 2), size=(720, 640), xguide="")
end

# ╔═╡ 8f6c1681-814e-4d00-aa43-3348eb388fd3
let p = plot(src; colorbar=false, title="SpectralPeak (loudest bin) on the $source_kind")
	scatter!(p, get_times(src), get_data(SpectralPeak(src)); ms=2.5, c=:white, msw=0, label="peak frequency")
end

# ╔═╡ b15845c9-5d6c-446e-8cd7-f010a00e2fe8
md"""
## 6. Phase-based deviations

These read the **phase** of the complex coefficients (`get_complex`), so they need a front end that has them: `Stft` or `Cwt`.
A `LinSpec` or `MelSpec` has discarded the phase; when one of those is selected above, this section uses the underlying `Stft` instead.
For a steady sinusoid the phase advances linearly, so its second difference `φ_t - 2φ_{t-1} + φ_{t-2}` is near zero; an onset or a noisy bin breaks that prediction.

| descriptor | formula |
|:--|:--|
| `SpectralPd` | mean over bins of `|φ_t - 2φ_{t-1} + φ_{t-2}|` (phase deviation). |
| `SpectralWpd` | the same, each bin weighted by its magnitude (quiet noisy bins stop dominating). |
| `SpectralNwpd` | `SpectralWpd` divided by the frame's mean magnitude (level independent). |
| `SpectralCd` | `Σ |X_t - X̂_t|`, `X̂_t = |X_{t-1}| e^{i(2φ_{t-1} - φ_{t-2})}`: magnitude and phase prediction error together. |
| `SpectralRcd` | `SpectralCd` over the bins whose magnitude increased only (onsets, not offsets). |

These descriptors have no keywords; their behaviour is set by the source (hop, window, wavelet grid).
A smaller hop makes the phase prediction more accurate, so steady parts get quieter and onsets stand out more.
"""

# ╔═╡ e0638e06-ec51-41b4-a94d-951f2a7c820e
phase_src = src isa Union{Stft, Cwt} ? src : base;

# ╔═╡ ed1cb6cb-6638-4b56-b64b-455cdefa4a9f
let ds = [SpectralPd(phase_src), SpectralWpd(phase_src), SpectralNwpd(phase_src),
          SpectralCd(phase_src), SpectralRcd(phase_src)]
	plot([plot(d; c=i, legend=false, yguide="", title=string(nameof(typeof(d)))) for (i, d) in enumerate(ds)]...;
	     layout=(5, 1), size=(720, 700), xguide="", plot_title="computed on $(nameof(typeof(phase_src)))", plot_titlefontsize=10)
end

# ╔═╡ cf8d1483-6f93-438c-8ce7-3a2be6bc1894
md"""
## 7. Multi-value descriptors: spectral contrast and polynomial fit

These return a small matrix per frame rather than one value.

**`SpectralContrast(spec; nbands, fmin, quantile, linear)`** (librosa `spectral_contrast`) splits the spectrum into octave bands above `fmin` (plus one band below it) and reports, per band, peak minus valley: the mean of the top `quantile` of the sorted magnitudes minus the mean of the bottom `quantile`, in natural-log units.

| keyword | effect |
|:--|:--|
| `nbands` | number of octave bands above `fmin`; the result has `nbands + 1` rows. The top band must stay below Nyquist: `fmin·2^(nbands-1) ≤ sr/2`. |
| `fmin` | lower edge of the first octave; raises or lowers every band. |
| `quantile` | fraction of each band taken as "peak" and "valley". Small values compare the extremes; large values compare broad averages and shrink the contrast. |
| `linear` | `true` returns the raw peak − valley instead of the log ratio. |

**`PolyFeatures(spec; order)`** (librosa `poly_features`) fits a polynomial of degree `order` to each frame's spectrum against frequency; row 1 is the highest-degree coefficient. `order=1` gives a slope and intercept (a linear tilt), higher orders describe curvature.
"""

# ╔═╡ 2d7dbfb5-095d-4abd-9f22-b0dd6aa56b2c
md"""
contrast nbands: $(@bind sc_nbands Slider(1:7; default=6, show_value=true))
fmin: $(@bind sc_fmin Slider(50:25:500; default=200, show_value=true)) Hz
quantile: $(@bind sc_q Slider(0.01:0.01:0.49; default=0.02, show_value=true))
linear $(@bind sc_linear CheckBox(default=false))

poly order: $(@bind poly_order Slider(0:6; default=1, show_value=true))
"""

# ╔═╡ 648c6be3-14d0-436e-9598-1501291bd373
contrast = let max_nbands = floor(Int, log2(sr / 2 / sc_fmin)) + 1
	nb = min(sc_nbands, max_nbands)
	nb < sc_nbands && @info "nbands clipped to $nb so the top octave stays below Nyquist"
	SpectralContrast(src; nbands=nb, fmin=sc_fmin, quantile=sc_q, linear=sc_linear)
end;

# ╔═╡ edf9f2b1-1841-4694-8495-35c797e70297
plot(contrast; title="SpectralContrast on the $source_kind", size=(720, 280))

# ╔═╡ 6cc3206e-6ecb-4037-be2b-6396ba29126a
poly = PolyFeatures(src; order=poly_order);

# ╔═╡ d9bbefe8-a250-4346-a1c4-de2eb7f9c8e6
md"""
Frame to inspect: $(@bind poly_frame Slider(1:get_nframes(src); default=cld(get_nframes(src), 2), show_value=true))
"""

# ╔═╡ 3fc37ccd-c95b-4153-ba68-5e5d5bb4b818
let f = collect(get_freq(src)), S = get_spec(src)[:, poly_frame], c = get_spec(poly)[:, poly_frame]
	fit = [sum(c[d+1] * x^(poly_order - d) for d in 0:poly_order) for x in f]
	p1 = plot(f, S; label="frame $poly_frame", xguide="Hz", title="spectrum and its degree-$poly_order fit")
	plot!(p1, f, fit; lw=2, label="PolyFeatures fit")
	p2 = plot(poly; title="PolyFeatures coefficients (row 1 = highest degree)")
	plot(p1, p2; layout=(2, 1), size=(720, 500))
end

# ╔═╡ 14435b3c-b6ed-492f-9d48-f5ea098d637f
md"""
The fit is done on the source values as they are (linear power or magnitude), so it is dominated by the loudest bins; it is most informative on a `MelSpec` with a magnitude spectrum.
"""

# ╔═╡ 9d42b0bd-0860-4cd5-ac06-b25edaabab24
md"""
## 8. Descriptor explorer

Tick any descriptors to compare them on one time axis, using the keyword values chosen above.
Each curve is min–max scaled to [0, 1] so descriptors with different units can share the plot.
"""

# ╔═╡ 3382947c-65a6-4dd1-8402-1913e1df56f6
descriptors = [
	"SpectralCentroid"  => s -> SpectralCentroid(s),
	"SpectralSpread"    => s -> SpectralSpread(s),
	"SpectralBandwidth" => s -> SpectralBandwidth(s; p=bw_p, normalize=bw_norm),
	"SpectralRolloff"   => s -> SpectralRolloff(s; threshold=rolloff_th),
	"SpectralSkewness"  => s -> SpectralSkewness(s),
	"SpectralKurtosis"  => s -> SpectralKurtosis(s),
	"SpectralFlatness"  => s -> SpectralFlatness(s),
	"SpectralEntropy"   => s -> SpectralEntropy(s; normalize=ent_norm),
	"SpectralCrest"     => s -> SpectralCrest(s),
	"SpectralDecrease"  => s -> SpectralDecrease(s),
	"SpectralSlope"     => s -> SpectralSlope(s),
	"SpectralFlux"      => s -> SpectralFlux(s; p=flux_p, step=flux_step, positive=flux_pos, root=flux_root, mean=flux_mean),
	"SpectralSd"        => s -> SpectralSd(s; step=flux_step, positive=flux_pos),
	"SpectralSf"        => s -> SpectralSf(s; step=flux_step, positive=flux_pos),
	"SpectralMkl"       => s -> SpectralMkl(s; mean=mkl_mean),
	"SpectralBroadband" => s -> SpectralBroadband(s; threshold=bb_th),
	"SpectralNovelty"   => s -> SpectralNovelty(s; step=flux_step, threshold=nov_th, method=nov_method, data=nov_data),
	"SpectralEnergy"    => s -> SpectralEnergy(s; log=en_log, gamma=en_gamma),
	"SpectralRms"       => s -> SpectralRms(s),
	"SpectralHfc"       => s -> SpectralHfc(s),
	"SpectralMax"       => s -> SpectralMax(s),
	"SpectralPeak"      => s -> SpectralPeak(s),
	"SpectralMean"      => s -> SpectralMean(s),
	"SpectralVar"       => s -> SpectralVar(s),
	"SpectralEef"       => s -> SpectralEef(s; normalize=ee_norm),
	"SpectralEer"       => s -> SpectralEer(s; normalize=ee_norm, gamma=eer_gamma),
	"SpectralPd"        => _ -> SpectralPd(phase_src),
	"SpectralWpd"       => _ -> SpectralWpd(phase_src),
	"SpectralNwpd"      => _ -> SpectralNwpd(phase_src),
	"SpectralCd"        => _ -> SpectralCd(phase_src),
	"SpectralRcd"       => _ -> SpectralRcd(phase_src),
];

# ╔═╡ 22b139c2-80d1-405f-b2e0-49a14046242f
@bind chosen MultiCheckBox(first.(descriptors); default=["SpectralCentroid", "SpectralFlatness", "SpectralFlux", "SpectralEnergy"])

# ╔═╡ d3a7bab4-b128-4093-85ec-bff97d8a6400
let p = plot(; xguide="Time (s)", yguide="scaled value", legend=:outerright, size=(720, 320))
	for (name, f) in descriptors
		name in chosen || continue
		d = f(src)
		v = Float64.(get_data(d))
		lo, hi = extrema(filter(isfinite, v); init=(0.0, 1.0))
		plot!(p, collect(get_times(d)), (v .- lo) ./ max(hi - lo, eps()); label=replace(name, "Spectral" => ""))
	end
	p
end

# ╔═╡ 3d28efe1-300f-469e-8adc-9fcab57761db
md"""
## Next

* **5 — Time–frequency transforms**: CWT, CQT, the S-transform, NSGT, synchrosqueezing, reassignment and the Cohen class; every descriptor here runs on those front ends too.
* **7 — Music and rhythm**: onset strength, novelty curves and tempo built on the flux and novelty descriptors of section 4.
* **8 — Time domain and pitch**: `Rms`, `Energy`, `Zcr` and pitch estimators, the frame-domain counterparts of section 5.

The full tour: 01_loading_and_frames, 02_stft_and_filterbanks, 03_cepstra, 04_spectral_descriptors, 05_time_frequency, 06_discrete_wavelets_and_decompositions, 07_music_and_rhythm, 08_time_domain_and_pitch, 09_signal_processing_and_classic.
"""

# ╔═╡ Cell order:
# ╟─a5bf0961-daa0-4af0-8782-7d07089eadb1
# ╟─e09a9adb-5474-41cf-8737-c5b375eb76fc
# ╟─a642ed41-4014-4927-bc50-232dea7859a7
# ╟─cbe30f27-1a3c-493f-acd8-5d8d63dc584d
# ╠═f803212b-1860-4c3b-ab70-5c66687fdef7
# ╠═e74f33fa-f009-4dde-9bdf-f20e02f34095
# ╟─4e9d4e11-ebd7-4692-8271-5136a92a861a
# ╟─c46eddc8-74f6-4403-8bfa-740a8f552c2c
# ╠═27e12e5e-9508-4fec-91f7-f80a167059b8
# ╠═a0e21baa-58ff-48b5-bd7f-1cf4e7af7b9a
# ╟─7d74db85-4a98-464b-8180-7b0e7ef39f9f
# ╠═85562057-4b9c-481e-b8af-2bc5f56b060f
# ╠═9cbd5df4-735c-42c5-af7d-58a1e6fa228d
# ╠═32d096c9-dd08-4a65-a874-452e661ecdf3
# ╟─ef5a7597-1a38-481c-ae38-e38adfe1b9b5
# ╟─0df65504-5f59-4adb-a97c-057a49bc8df7
# ╠═d5234530-1050-4f78-91af-dc7bd4f90c8e
# ╟─cdfb7701-b62e-42e8-97a9-4612fc0ec2f7
# ╟─5378e434-b13c-4eaa-8ea7-1bbe1ec6f41b
# ╟─90981d4a-862d-41ca-a572-c0700beb3f02
# ╠═55d6d7b9-ebc5-4559-b06b-a593063acebc
# ╟─901c6812-76c4-46d0-a027-e5f306507e6a
# ╟─0b5591d4-c5ce-4a9d-88d7-8c1953a42a40
# ╠═80eccd0f-70b4-4b99-8cd1-d139faa6903a
# ╠═8f6c1681-814e-4d00-aa43-3348eb388fd3
# ╟─b15845c9-5d6c-446e-8cd7-f010a00e2fe8
# ╠═e0638e06-ec51-41b4-a94d-951f2a7c820e
# ╠═ed1cb6cb-6638-4b56-b64b-455cdefa4a9f
# ╟─cf8d1483-6f93-438c-8ce7-3a2be6bc1894
# ╟─2d7dbfb5-095d-4abd-9f22-b0dd6aa56b2c
# ╠═648c6be3-14d0-436e-9598-1501291bd373
# ╠═edf9f2b1-1841-4694-8495-35c797e70297
# ╠═6cc3206e-6ecb-4037-be2b-6396ba29126a
# ╟─d9bbefe8-a250-4346-a1c4-de2eb7f9c8e6
# ╠═3fc37ccd-c95b-4153-ba68-5e5d5bb4b818
# ╟─14435b3c-b6ed-492f-9d48-f5ea098d637f
# ╟─9d42b0bd-0860-4cd5-ac06-b25edaabab24
# ╠═3382947c-65a6-4dd1-8402-1913e1df56f6
# ╠═22b139c2-80d1-405f-b2e0-49a14046242f
# ╠═d3a7bab4-b128-4093-85ec-bff97d8a6400
# ╟─3d28efe1-300f-469e-8adc-9fcab57761db
