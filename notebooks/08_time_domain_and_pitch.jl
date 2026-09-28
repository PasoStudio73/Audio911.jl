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

# ╔═╡ d78b3785-69f9-484c-9ea9-33f556a7b38b
begin
	import Pkg
	# the notebook environment next to this file pins Audio911 (from this repository), PlutoUI and Plots
	Pkg.activate(@__DIR__)
	Pkg.instantiate()
	using Audio911, PlutoUI, Plots
	include(joinpath(@__DIR__, "common.jl"))
	gr(); default(size=(720, 300), legend=:topright, titlefontsize=10, guidefontsize=9)
end

# ╔═╡ d04829c0-0c4a-40de-b47d-6f930c2ff801
md"""
# Audio911 · 8 — Time-domain features and pitch

The features in this notebook read the **frames directly**, without any transform:
level (`Rms`, `Energy`), zero crossings (`Zcr`, `zero_crossings`, `Ezr`), periodicity (`autocorrelate`, `HarmonicRatio`), and the fundamental frequency (`Pitch`) with seven interchangeable estimators:

| family | estimators |
|:--|:--|
| time domain | `pitch_ncf` (MATLAB default), `pitch_yin` (librosa `yin`) |
| cepstral | `pitch_cep` |
| spectral (ported from audioFlux) | `pitch_pef`, `pitch_hps`, `pitch_lhs`, `pitch_stft` |

Every result is an `AbstractSpectral` with one value per frame; `get_times` gives the same time axis as every spectrogram built on the same `Frames`, so tracks can be drawn straight on top of an `Stft`.

To judge accuracy, the signal menu adds two synthetic voices built with `synth_f0`, whose true f0 contour is known and drawn as a dashed line.
"""

# ╔═╡ a257ddc9-980d-4c67-ab71-0c71275de5b1
TableOfContents()

# ╔═╡ f5e78a72-03ad-498d-92e5-a743e6bf3572
md"""
## 1. The signal

Besides the shared menu, two synthetic voices are available.
Each is a sum of harmonics `h·f0(t)` with amplitude `1/h`, synthesised with `synth_f0(times, frequencies, sr; amplitudes)`, which linearly interpolates a frequency (and amplitude) curve given at breakpoints and integrates the phase sample by sample.

| synthetic control | effect |
|:--|:--|
| base f0 | fundamental of the voice |
| vibrato depth / rate | periodic f0 modulation; fast deep vibrato is hard for long frames, which average the pitch over their length |
| glide | an octave jump and a slide in the middle of the "glide" voice |
| harmonics | 1 gives a pure sine (easy for time-domain methods, hard for harmonic-sieve methods); many harmonics look like a real voice |
| drop fundamental | removes harmonic 1: the ear still hears f0 ("missing fundamental"); methods that only look for the strongest peak fail |
| noise | white noise level relative to the voice |
"""

# ╔═╡ 6ed30e3c-810e-48fb-971e-e235213846ce
md"""
Signal: $(@bind signal_name Select(["vibrato" => "Synthetic voice with vibrato (known f0)", "glide" => "Synthetic voice with glide and octave jump (known f0)", SIGNALS...]))

base f0 (Hz): $(@bind f0_base Slider(80:5:400; default=150, show_value=true))
vibrato depth (%): $(@bind vib_depth Slider(0:0.5:10; default=3, show_value=true))
vibrato rate (Hz): $(@bind vib_rate Slider(1:0.5:10; default=5, show_value=true))

harmonics: $(@bind n_harm Slider(1:12; default=6, show_value=true))
drop fundamental $(@bind drop_f0 CheckBox(default=false))
noise level: $(@bind noise Slider(0:0.02:1; default=0.05, show_value=true))
"""

# ╔═╡ 14e8a4f1-19af-4796-bbe8-bd4da8710959
sr = 16000

# ╔═╡ 4f82f0cd-b224-439a-a1a9-dd88cec1daf7
begin
	# true f0 contour of the synthetic voices, sampled every 5 ms
	truth_t = collect(0:0.005:2.0)
	truth_f = if signal_name == "vibrato"
		f0_base .* (1 .+ vib_depth / 100 .* sin.(2π * vib_rate .* truth_t))
	elseif signal_name == "glide"
		base = [t < 0.7 ? f0_base : t < 1.0 ? 2f0_base : f0_base * (2 - (t - 1.0)) for t in truth_t]
		base .* (1 .+ vib_depth / 100 .* sin.(2π * vib_rate .* truth_t))
	else
		nothing
	end
end;

# ╔═╡ 0f9f741a-789e-4afe-943c-ef5193f3bbd9
audio = if isnothing(truth_f)
	demo_signal(signal_name; sr)
else
	let hs = (drop_f0 ? 2 : 1):n_harm
		# harmonics above Nyquist are silenced through their amplitude
		x = sum(synth_f0(truth_t, h .* truth_f, sr;
		                 amplitudes=[h * f < sr / 2 ? 1 / h : 0.0 for f in truth_f]) for h in hs)
		x = x ./ maximum(abs, x) .+ noise .* randn(length(x)) ./ 3
		AudioFile(Float32.(x ./ maximum(abs, x)), sr)
	end
end

# ╔═╡ ab784382-969a-429d-b513-eb00fb7fd81f
plot(audio)

# ╔═╡ f3c229d4-7800-44e6-b1aa-e5a9130264f5
audio_player(audio)

# ╔═╡ 1e09b077-e46e-4be7-a225-3ac425c84249
md"""
## 2. Frames

All features below share these frames. Pitch needs frames holding **at least two periods** of the lowest f0 you search for (`pitch_yin` needs the lag range inside half the frame), so it usually wants longer frames than a spectrogram: 1024 samples at 16 kHz is 64 ms, enough for 30 Hz with NCF and 60 Hz with YIN.
"""

# ╔═╡ 6dc2024a-a33a-432c-81f1-53ffd5e2d090
md"""
winsize: $(@bind winsize Select([256, 512, 1024, 2048]; default=1024))
hop: $(@bind winstep Select([64, 128, 256, 512]; default=256))
window: $(@bind wintype Select([hanning => "hanning", hamming => "hamming", rect => "rect", blackman => "blackman"]))
"""

# ╔═╡ 71ff9808-d127-4ea8-b8ed-b9b2e092456d
frames = Frames(audio; winsize, winstep, type=wintype)

# ╔═╡ 49679f9d-4068-4c7d-80a6-59eafd3f4bf6
times = get_times(Rms(frames));

# ╔═╡ dba35c58-ba5c-444b-8bbc-16180002c6a3
md"""
## 3. Level: `Rms` and `Energy`

| call | parameter | effect |
|:--|:--|:--|
| `Rms(frames; windowed=false)` | `windowed` | `sqrt(mean(x.^2))` of every frame (librosa `rms`). `true` applies the analysis window first, which lowers the value (a Hann window keeps about 61% of the RMS) and smooths the curve, because samples at the frame edges count less. |
| `Rms(spec)` | — | the same level estimated from a power `Stft` through Parseval's theorem (librosa `rms(S=...)`); it sees the windowed frame, so it matches `windowed=true` up to window scaling. |
| `Energy(frames; windowed=true)` | `windowed` | short-time energy `Σ x.^2` (MATLAB `shortTimeEnergy`), grows with the frame length. `false` is the raw frame energy, the same as `get_energy(frames)`. |

RMS: windowed $(@bind rms_windowed CheckBox(default=false))
Energy: windowed $(@bind en_windowed CheckBox(default=true))
"""

# ╔═╡ 0e04c3fe-ef59-4275-b580-70f10828cbff
begin
	rms  = Rms(frames; windowed=rms_windowed)
	rmsS = Rms(Stft(frames))
	en   = Energy(frames; windowed=en_windowed)
	p1 = plot(times, get_data(rms); label="Rms(frames; windowed=$rms_windowed)", title="RMS", yguide="linear")
	plot!(p1, times, get_data(rmsS); label="Rms(Stft(frames))", ls=:dash)
	p2 = plot(times, 10 .* log10.(get_data(en) .+ 1f-10); label="Energy(frames; windowed=$en_windowed)",
	          title="energy", yguide="dB", xguide="Time (s)", c=3)
	plot(p1, p2; layout=(2, 1), size=(720, 420))
end

# ╔═╡ 34f872e4-2b47-4a6e-88b6-fa3ad40a6bbb
md"""
## 4. Zero crossings: `Zcr`, `zero_crossings` and `Ezr`

Noise-like sounds (fricatives, breath, cymbals) cross zero often; voiced sounds rarely.
A crossing rate is therefore a cheap voiced/unvoiced cue, and the energy-to-crossing ratio `Ezr` separates the two even better.

| call | parameter | effect |
|:--|:--|:--|
| `Zcr(frames; ...)` | `threshold=1e-10` | samples with `|x| ≤ threshold` count as zero (librosa); raising it ignores low-level noise wiggling around zero, so silence stops producing crossings. |
| | `rate=true` | crossings divided by the frame length; `false` returns raw counts. |
| | `windowed=false` | count on the windowed frame (audioFlux `Temporal`). |
| | `strict=false` | count only strict sign changes `x[i]·x[i-1] < 0`, so exact zeros never cross (audioFlux). |
| `zero_crossings(x; threshold)` | `threshold` | per-sample Boolean mask of the sign changes (librosa `zero_crossings`). |
| `Ezr(frames; gamma=1)` | `gamma` | `log10(1 + γE) / (Z + 1)` on the windowed frame (audioFlux `Temporal.ezr`); `γ` scales the energy before the log, so large `γ` compresses loud frames and highlights quiet voiced ones. |
"""

# ╔═╡ e2497256-8259-4f7a-93d9-efe560190f93
md"""
threshold (log10): $(@bind zcr_logth Slider(-10:0.5:-1; default=-10, show_value=true))
rate $(@bind zcr_rate CheckBox(default=true))
windowed $(@bind zcr_windowed CheckBox(default=false))
strict $(@bind zcr_strict CheckBox(default=false))
Ezr γ (log10): $(@bind ezr_loggamma Slider(-2:0.5:4; default=0, show_value=true))
"""

# ╔═╡ d4d7e65b-efaf-4381-97c7-92b192c937d1
begin
	zcr = Zcr(frames; threshold=10.0^zcr_logth, rate=zcr_rate, windowed=zcr_windowed, strict=zcr_strict)
	ezr = Ezr(frames; gamma=10.0^ezr_loggamma)
	q1 = plot(times, get_data(zcr); label="Zcr", title="zero-crossing $(zcr_rate ? "rate" : "count")")
	q2 = plot(times, get_data(ezr); label="Ezr (γ = 10^$ezr_loggamma)", c=2, title="energy / zero-crossing ratio", xguide="Time (s)")
	plot(q1, q2; layout=(2, 1), size=(720, 420))
end

# ╔═╡ 649f205a-b75b-488d-b0c9-adbef85ef47d
md"""
`zero_crossings` on a 20 ms excerpt, starting at: $(@bind zc_start Slider(0:0.01:1.9; default=0.5, show_value=true)) s
"""

# ╔═╡ df52ca8f-397e-4bf6-92df-fc6ea2f5ab6b
let x = vec(get_data(audio)), i0 = round(Int, zc_start * sr) + 1, n = round(Int, 0.02sr)
	seg = x[i0:min(end, i0 + n)]
	z = zero_crossings(seg; threshold=10.0^zcr_logth)
	t = (i0 - 1 .+ (0:length(seg)-1)) ./ sr
	plot(t, seg; label="signal", xguide="Time (s)", title="$(count(z)) crossings in 20 ms")
	scatter!(t[z], zeros(count(z)); label="zero_crossings", ms=3)
end

# ╔═╡ 08d0eff9-8de2-46ff-b052-ae8d08151aea
md"""
## 5. Periodicity: `autocorrelate` and `HarmonicRatio`

`autocorrelate(x; max_size)` returns `r[k] = Σ x[n]·x[n+k]` for `k = 0:max_size-1`, computed with an FFT (librosa `autocorrelate`).
A periodic frame has peaks at multiples of its period; the first strong one gives f0 = sr / lag, which is the basis of `pitch_ncf` and `pitch_yin`.
`max_size` only truncates the returned lags (keep it above `sr / fmin`).

Frame: $(@bind ac_frame Slider(1:length(frames); default=cld(length(frames), 3), show_value=true))
max lag (ms): $(@bind ac_maxms Slider(5:1:40; default=20, show_value=true))
"""

# ╔═╡ 4583ec17-a33b-4748-a70e-3ce2d0778243
let seg = get_data(frames)[:, ac_frame]
	m = min(length(seg), round(Int, ac_maxms * sr / 1000))
	r = autocorrelate(seg; max_size=m)
	lags = (0:length(r)-1) ./ sr .* 1000
	plot(lags, r ./ r[1]; label="r[k] / r[0]", xguide="lag (ms)", title="autocorrelation of frame $ac_frame")
	if !isnothing(truth_f)
		f = truth_f[argmin(abs.(truth_t .- times[ac_frame]))]
		vline!(1000 .* (1:4) ./ f; label="true period × 1…4", ls=:dash)
	end
	plot!()
end

# ╔═╡ e0a9b4d9-198e-4916-98c8-d9c2822eea6d
md"""
`HarmonicRatio` is the maximum normalised autocorrelation over the lags of a pitch range: 1 for a perfectly periodic frame, near 0 for noise.

| parameter | effect |
|:--|:--|
| `range=(50, 400)` | lag search limits `sr/hi … sr/lo`. A range that misses the true f0 still finds a peak at a multiple of the period, so the ratio stays high for voiced frames. |
| `method=:matlab` | MATLAB `harmonicRatio`: raw frames, normalised autocorrelation. |
| `method=:audioflux` | audioFlux `HarmonicRatio`: windowed frames zero-padded to twice their length, search from the first zero crossing of the autocorrelation up to `sr / fmin`, quadratic peak refinement. |
| `fmin=range[1]` | lowest frequency (longest lag) of the `:audioflux` search. |

method: $(@bind hr_method Select([:matlab, :audioflux]))
range low: $(@bind hr_lo Slider(30:10:300; default=50, show_value=true))
range high: $(@bind hr_hi Slider(200:50:2000; default=400, show_value=true))
fmin (`:audioflux`): $(@bind hr_fmin Slider(30:10:300; default=50, show_value=true))
"""

# ╔═╡ d4763920-f042-4ed9-a200-aebf8d81f33f
let hr = HarmonicRatio(frames; range=(hr_lo, hr_hi), method=hr_method, fmin=hr_fmin)
	plot(times, get_data(hr); label="HarmonicRatio(:$hr_method)", ylims=(0, 1.05), xguide="Time (s)",
	     title="harmonic ratio")
	plot!(times, get_data(rms) ./ maximum(get_data(rms)); label="RMS (normalised)", ls=:dot)
end

# ╔═╡ 2da691ee-2710-4238-ad83-3ad0b408caa6
md"""
## 6. Pitch

`Pitch(frames; method, range, threshold)` runs a per-frame estimator on every frame and returns f0 in Hz, `0` for frames judged unvoiced.
Any function `(x, sr; range, threshold) -> f0` is a valid `method`, so the extra keywords of the spectral estimators are passed with a closure, e.g.
`method = (x, sr; kw...) -> pitch_hps(x, sr; harmonics=3, kw...)`.

| keyword | effect |
|:--|:--|
| `range=(50, 400)` | search limits in Hz. Too wide invites octave errors (a period multiple or a harmonic wins); too narrow clips the track at its edges. It must be resolvable: `sr / range[1]` must fit in the frame (half the frame for YIN). |
| `threshold` | method-specific voicing / peak threshold, see the table below; `nothing` picks the method's default (`0.1` for YIN, `20` dB for `pitch_stft`, `0` otherwise). |

| estimator | how it works | its own keywords |
|:--|:--|:--|
| `pitch_ncf` | normalised correlation `r[k] / sqrt(r[0]·e[k])`, best lag in `range`, parabolic refinement (MATLAB `"NCF"`). Robust, prone to octave-down errors on strongly periodic envelopes. | `threshold`: minimum correlation; frames below it return 0 (voicing decision). |
| `pitch_yin` | cumulative-mean-normalised difference; first dip below `threshold`, else the global minimum (de Cheveigné & Kawahara). Very accurate on clean voices. | `threshold=0.1`: lower is stricter and prefers the *first* dip, avoiding octave-down errors less often chosen by noise. |
| `pitch_cep` | highest real-cepstrum peak between the quefrencies of `range` (MATLAB `"CEP"`). Needs several harmonics: fails on pure sines. | `threshold`: minimum cepstral peak height. |
| `pitch_pef` | power spectrum on a log-frequency grid cross-correlated with a comb-like filter `1/(γ - cos 2πq)` (Gonzalez & Brookes, MATLAB / audioFlux `"PEF"`). Slowest; good with missing fundamentals. | `cutoff=4000` top of the log grid; `alpha=10`, `beta=0.5` span of the filter (`q` from `β` to `α+β`, i.e. about `α` harmonics); `gamma=1.8` peak sharpness (closer to 1 = narrower teeth); `window=hamming`. |
| `pitch_hps` | harmonic product `Π |X[k·j]|` over `harmonics` multiples (audioFlux `PitchHPS`). About 1 Hz bins; always returns a frequency. | `harmonics=5`: more harmonics suppress octave-up errors but fail if upper harmonics are weak; `window=hamming`. |
| `pitch_lhs` | the same with the sum of log magnitudes (Hermes, MATLAB `"LHS"`); less dominated by one strong partial. | `harmonics=5`, `window=hamming`. |
| `pitch_stft` | spectral peaks within `threshold` dB of the strongest, candidates `f`, `f/2`, `f/3` scored by a harmonic sieve. | `threshold=20` dB peak window (larger admits weak harmonics and noise); `tolerance=0.03` relative harmonic match; `window=hamming`. |
"""

# ╔═╡ 473eecc2-507d-4825-b54b-e8b4452cb28c
md"""
### Common search range

range low (Hz): $(@bind p_lo Slider(30:5:300; default=60, show_value=true))
range high (Hz): $(@bind p_hi Slider(200:10:1200; default=500, show_value=true))
"""

# ╔═╡ e56589b7-14cf-45df-b600-5b428b6a7bb5
prange = (p_lo, max(p_hi, p_lo + 20))

# ╔═╡ b99bdd71-f8cd-4b84-87b1-115c4560f759
md"""
### Per-estimator controls

**NCF** threshold: $(@bind th_ncf Slider(0:0.05:0.95; default=0.0, show_value=true))
**YIN** threshold: $(@bind th_yin Slider(0.02:0.02:0.6; default=0.1, show_value=true))
**CEP** threshold: $(@bind th_cep Slider(0:0.01:0.3; default=0.0, show_value=true))

**PEF** cutoff: $(@bind pef_cutoff Slider(1000:500:7500; default=4000, show_value=true))
α: $(@bind pef_alpha Slider(2:1:20; default=10, show_value=true))
β: $(@bind pef_beta Slider(0.1:0.1:1.0; default=0.5, show_value=true))
γ: $(@bind pef_gamma Slider(1.1:0.1:3.0; default=1.8, show_value=true))

**HPS / LHS** harmonics: $(@bind hps_h Slider(1:10; default=5, show_value=true))
window: $(@bind hps_win Select([hamming => "hamming", hanning => "hanning", rect => "rect", blackman => "blackman"]))

**STFT sieve** threshold (dB): $(@bind th_stft Slider(5:5:60; default=20, show_value=true))
tolerance: $(@bind stft_tol Slider(0.005:0.005:0.1; default=0.03, show_value=true))
"""

# ╔═╡ b914ef75-119d-4ab1-b61e-57725f4559f8
f0_ncf = Pitch(frames; method=pitch_ncf, range=prange, threshold=th_ncf);

# ╔═╡ d0ba2b99-f9bd-4f0e-b22e-7c0deb81ebe7
f0_yin = Pitch(frames; method=pitch_yin, range=prange, threshold=th_yin);

# ╔═╡ 22b52701-d915-4ea1-8b37-6ea8f7aa7ac2
f0_cep = Pitch(frames; method=pitch_cep, range=prange, threshold=th_cep);

# ╔═╡ 02ca4e7d-a0d8-41ec-906d-767aee1de69b
f0_pef = Pitch(frames; range=prange,
               method=(x, sr; kw...) -> pitch_pef(x, sr; cutoff=pef_cutoff, alpha=pef_alpha,
                                                  beta=pef_beta, gamma=pef_gamma, kw...));

# ╔═╡ c5bb9e51-b316-49e9-9b3d-8ed96c06a3f2
f0_hps = Pitch(frames; range=prange,
               method=(x, sr; kw...) -> pitch_hps(x, sr; harmonics=hps_h, window=hps_win, kw...));

# ╔═╡ 21d5926c-e5dc-4e6b-ab40-ff1829b63f50
f0_lhs = Pitch(frames; range=prange,
               method=(x, sr; kw...) -> pitch_lhs(x, sr; harmonics=hps_h, window=hps_win, kw...));

# ╔═╡ e5550022-bace-4a24-97e1-94db5267380b
f0_stft = Pitch(frames; range=prange, threshold=th_stft,
                method=(x, sr; kw...) -> pitch_stft(x, sr; tolerance=stft_tol, kw...));

# ╔═╡ 2f1b49e0-68d5-4a16-a6a1-61efdc2d0901
tracks = ["ncf" => f0_ncf, "yin" => f0_yin, "cep" => f0_cep, "pef" => f0_pef,
          "hps" => f0_hps, "lhs" => f0_lhs, "stft" => f0_stft];

# ╔═╡ e0f5104c-fe99-4378-8db9-4ce136909b04
md"""
### All tracks on the spectrogram

Show: $(@bind shown MultiCheckBox(first.(tracks); default=["ncf", "yin", "pef", "hps"]))
top of plot (Hz): $(@bind fmax_plot Slider(300:100:4000; default=1000, show_value=true))
"""

# ╔═╡ 3e454f86-b4bf-41bc-a349-644293fe8ce3
begin
	# unvoiced frames (0 Hz) are drawn as gaps
	gap(v) = [x > 0 ? x : NaN for x in v]
	truth_at(t) = isnothing(truth_f) ? NaN : truth_f[argmin(abs.(truth_t .- t))]
	plot(Stft(frames; nfft=max(2048, winsize)); ylims=(0, fmax_plot), top_db=60, c=:grays, colorbar=false,
	     title="pitch tracks, range $prange Hz", size=(720, 380))
	isnothing(truth_f) || plot!(truth_t, truth_f; label="true f0", c=:red, lw=1.5, ls=:dash)
	for (name, p) in tracks
		name in shown && plot!(times, gap(get_data(p)); label=name, lw=2)
	end
	plot!(legend=:outertopright)
end

# ╔═╡ 37d34bac-a390-46db-8b42-8a93979c6328
md"""
### Accuracy against the known contour

For the synthetic voices, each estimator is scored on every frame: **gross errors** are frames more than 50 cents (half a semitone) away from the truth, typically octave jumps; **median error** is taken over the remaining frames; **voiced** is the fraction of frames with a non-zero estimate.
Try *drop fundamental*, one harmonic, heavy noise or deep fast vibrato to see which methods break first.
"""

# ╔═╡ 4d172e4f-8b2c-4630-b873-04020da2c24f
if isnothing(truth_f)
	md"_Choose one of the synthetic voices to compute accuracy._"
else
	let truth = truth_at.(times)
		rows = map(tracks) do (name, p)
			f = get_data(p)
			v = f .> 0
			cents = 1200 .* abs.(log2.(f[v] ./ truth[v]))
			good = cents .≤ 50
			(estimator = name,
			 voiced = round(count(v) / length(f); digits=2),
			 gross_error_pct = round(100 * count(!, good) / max(1, count(v)); digits=1),
			 median_cents = isempty(cents[good]) ? NaN : round(sort(cents[good])[cld(count(good), 2)]; digits=1))
		end
		rows
	end
end

# ╔═╡ b31bc3ad-cdee-40a5-857a-295cc87af69a
md"""
### One frame, every estimator

Frame: $(@bind pf Slider(1:length(frames); default=cld(length(frames), 2), show_value=true))
"""

# ╔═╡ 1e22c252-24b5-4ebf-bd47-8d7bdc2e9c46
let x = get_data(frames)[:, pf]
	est = [(name, get_data(p)[pf]) for (name, p) in tracks]
	f = truth_at(times[pf])
	X = abs.(Audio911.FFTW.rfft(vcat(x .* get_window(frames), zeros(Float32, 8192 - length(x)))))
	fr = range(0, sr / 2; length=length(X))
	plot(fr, 20 .* log10.(X ./ maximum(X) .+ 1f-6); label="frame spectrum", xlims=(0, fmax_plot),
	     xguide="Hz", yguide="dB", title="frame $pf at $(round(times[pf]; digits=3)) s", ylims=(-80, 3))
	isnan(f) || vline!([f]; label="true f0 = $(round(f; digits=1))", lw=3, ls=:dash)
	for (i, (name, v)) in enumerate(est)
		v > 0 && vline!([v]; label="$name = $(round(v; digits=1))", c=i + 1)
	end
	plot!(legend=:outertopright)
end

# ╔═╡ 4c125651-bf0b-48d8-b3ad-2bc8ec45cb66
md"""
### Cost

Time per call on the current frames, which is also the reason `pitch_pef` gets its own cell above: dragging its sliders does not recompute the other six tracks.
"""

# ╔═╡ aebd62ea-df74-41d6-8f72-c1d368399bff
[(name, round(1000 * @elapsed(Pitch(frames; method=m, range=prange)); digits=1))
 for (name, m) in ["ncf" => pitch_ncf, "yin" => pitch_yin, "cep" => pitch_cep, "pef" => pitch_pef,
                   "hps" => pitch_hps, "lhs" => pitch_lhs, "stft" => pitch_stft]]

# ╔═╡ ad343392-bc0f-4f47-8ba2-25b55c7f84d3
md"""
### Listening to a track

`synth_f0(times, frequencies, sr; amplitudes)` also turns an *estimated* track back into sound: here the chosen estimator's f0 is resynthesised as a sine whose amplitude follows the frame RMS (silenced on unvoiced frames). Compare it with the original to hear octave errors immediately.

Estimator: $(@bind listen_name Select(first.(tracks); default="yin"))
"""

# ╔═╡ e767126b-d91d-40fb-82b6-3061b2973507
let p = Dict(tracks)[listen_name], f = get_data(p)
	amp = get_data(Rms(frames)) .* (f .> 0)
	fill_f = [x > 0 ? x : 100f0 for x in f]           # keep the phase running through gaps
	y = synth_f0(vcat(0, times), vcat(fill_f[1], fill_f), sr; amplitudes=vcat(0, amp))
	audio_player(y, sr)
end

# ╔═╡ 74754f71-cf58-4784-a965-c425a27a8ef0
md"""
## Next

* **9 — Signal processing and classic methods**: unit conversions, decibels and weighting, synthesis, silence trimming, LPC, Hilbert / CZT / correlation, time stretch and pitch shift, NMF and HMM / Viterbi.

The full tour: 01 loading and frames · 02 STFT and filterbanks · 03 cepstra · 04 spectral descriptors · 05 time–frequency transforms · 06 discrete wavelets and decompositions · 07 music and rhythm · **08 time domain and pitch** · 09 signal processing and classic.
"""

# ╔═╡ Cell order:
# ╟─d04829c0-0c4a-40de-b47d-6f930c2ff801
# ╟─d78b3785-69f9-484c-9ea9-33f556a7b38b
# ╟─a257ddc9-980d-4c67-ab71-0c71275de5b1
# ╟─f5e78a72-03ad-498d-92e5-a743e6bf3572
# ╟─6ed30e3c-810e-48fb-971e-e235213846ce
# ╠═14e8a4f1-19af-4796-bbe8-bd4da8710959
# ╠═4f82f0cd-b224-439a-a1a9-dd88cec1daf7
# ╠═0f9f741a-789e-4afe-943c-ef5193f3bbd9
# ╠═ab784382-969a-429d-b513-eb00fb7fd81f
# ╠═f3c229d4-7800-44e6-b1aa-e5a9130264f5
# ╟─1e09b077-e46e-4be7-a225-3ac425c84249
# ╟─6dc2024a-a33a-432c-81f1-53ffd5e2d090
# ╠═71ff9808-d127-4ea8-b8ed-b9b2e092456d
# ╠═49679f9d-4068-4c7d-80a6-59eafd3f4bf6
# ╟─dba35c58-ba5c-444b-8bbc-16180002c6a3
# ╠═0e04c3fe-ef59-4275-b580-70f10828cbff
# ╟─34f872e4-2b47-4a6e-88b6-fa3ad40a6bbb
# ╟─e2497256-8259-4f7a-93d9-efe560190f93
# ╠═d4d7e65b-efaf-4381-97c7-92b192c937d1
# ╟─649f205a-b75b-488d-b0c9-adbef85ef47d
# ╠═df52ca8f-397e-4bf6-92df-fc6ea2f5ab6b
# ╟─08d0eff9-8de2-46ff-b052-ae8d08151aea
# ╠═4583ec17-a33b-4748-a70e-3ce2d0778243
# ╟─e0a9b4d9-198e-4916-98c8-d9c2822eea6d
# ╠═d4763920-f042-4ed9-a200-aebf8d81f33f
# ╟─2da691ee-2710-4238-ad83-3ad0b408caa6
# ╟─473eecc2-507d-4825-b54b-e8b4452cb28c
# ╠═e56589b7-14cf-45df-b600-5b428b6a7bb5
# ╟─b99bdd71-f8cd-4b84-87b1-115c4560f759
# ╠═b914ef75-119d-4ab1-b61e-57725f4559f8
# ╠═d0ba2b99-f9bd-4f0e-b22e-7c0deb81ebe7
# ╠═22b52701-d915-4ea1-8b37-6ea8f7aa7ac2
# ╠═02ca4e7d-a0d8-41ec-906d-767aee1de69b
# ╠═c5bb9e51-b316-49e9-9b3d-8ed96c06a3f2
# ╠═21d5926c-e5dc-4e6b-ab40-ff1829b63f50
# ╠═e5550022-bace-4a24-97e1-94db5267380b
# ╠═2f1b49e0-68d5-4a16-a6a1-61efdc2d0901
# ╟─e0f5104c-fe99-4378-8db9-4ce136909b04
# ╠═3e454f86-b4bf-41bc-a349-644293fe8ce3
# ╟─37d34bac-a390-46db-8b42-8a93979c6328
# ╠═4d172e4f-8b2c-4630-b873-04020da2c24f
# ╟─b31bc3ad-cdee-40a5-857a-295cc87af69a
# ╠═1e22c252-24b5-4ebf-bd47-8d7bdc2e9c46
# ╟─4c125651-bf0b-48d8-b3ad-2bc8ec45cb66
# ╠═aebd62ea-df74-41d6-8f72-c1d368399bff
# ╟─ad343392-bc0f-4f47-8ba2-25b55c7f84d3
# ╠═e767126b-d91d-40fb-82b6-3061b2973507
# ╟─74754f71-cf58-4784-a965-c425a27a8ef0
