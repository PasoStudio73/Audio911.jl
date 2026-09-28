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

# ╔═╡ 03e3cc2f-e47f-40e1-8ffc-fdfadedaadef
begin
	import Pkg
	# the notebook environment next to this file pins Audio911 (from this repository), PlutoUI and Plots
	Pkg.activate(@__DIR__)
	Pkg.instantiate()
	using Audio911, PlutoUI, Plots
	using Random: Xoshiro
	include(joinpath(@__DIR__, "common.jl"))
	gr(); default(size=(720, 300), legend=:topright, titlefontsize=10, guidefontsize=9)
end

# ╔═╡ 8bfb5dcf-f51b-4493-b6d3-712211a73619
md"""
# Audio911 · 9 — Signal utilities, DSP, time stretching, NMF and HMM

The last notebook of the tour collects the tools that sit *around* the feature pipeline:

* **unit conversions** between Hz, mel, MIDI, note names, frames, samples and seconds;
* **decibels and loudness weighting** (A, B, C, D curves);
* **signal utilities**: μ-law, normalisation, feature scaling, level summaries, synthesis, silence trimming and LPC;
* **DSP building blocks**: Hilbert envelope, chirp-Z zoom FFT, cross-correlation and convolution;
* **time stretching and pitch shifting** with a phase vocoder;
* two **classic models**: non-negative matrix factorisation and discrete hidden Markov models.
"""

# ╔═╡ aeae8f57-ea68-4e68-aa45-b296168bb48a
TableOfContents()

# ╔═╡ 898ec42b-2193-4982-8e48-c7094b95783a
md"""
Signal used throughout: $(@bind signal_name Select(SIGNALS))
"""

# ╔═╡ 613a1703-9714-4546-ad94-2089a6e8a842
begin
	audio = demo_signal(signal_name)
	x  = vec(get_data(audio))
	sr = get_sr(audio)
	audio_player(audio)
end

# ╔═╡ cca58518-0890-4837-8370-20b163f96b22
md"""
## 1. Unit conversions

| function | meaning |
|:--|:--|
| `hz_to_mel(f; htk)`, `mel_to_hz(m; htk)` | `htk=false`: Slaney's scale (linear below 1 kHz, logarithmic above, librosa's default); `htk=true`: `2595·log10(1 + f/700)`. |
| `hz_to_midi`, `midi_to_hz` | MIDI number `69 + 12·log2(f/440)`. |
| `midi_to_note(n; octave, cents)`, `hz_to_note(f; ...)` | note name; `octave=false` drops the octave digit, `cents=true` appends the deviation from the nearest semitone. |
| `note_to_midi`, `note_to_hz` | parse `"C4"`, `"F#3"`, `"Bb2"`. |
| `fft_frequencies(sr, nfft)` | centres of the one-sided FFT bins. |
| `mel_frequencies(n; fmin, fmax, htk)` | `n` frequencies equally spaced in mel. |
| `cqt_frequencies(n; fmin, bins_per_octave, tuning)` | geometric grid; `tuning` shifts it by a fraction of a bin. |
| `tempo_frequencies(n, hop, sr)` | BPM of autocorrelation lags `0:n-1`. |
| `frames_to_samples`, `samples_to_frames`, `frames_to_time`, `time_to_frames`, `samples_to_time`, `time_to_samples` | 1-based index ↔ time conversions; `offset` is the sample offset of frame 1. |
"""

# ╔═╡ 5914d8d3-fb68-4061-8cd8-3369fbd7937f
md"""
Frequency: $(@bind fconv Slider(20:1:8000; default=440, show_value=true)) Hz
cents $(@bind showcents CheckBox(default=true))
octave $(@bind showoct CheckBox(default=true))
"""

# ╔═╡ 70e2424e-29a0-4e93-a491-31ad38d54cd3
(mel_slaney = hz_to_mel(fconv), mel_htk = hz_to_mel(fconv; htk=true),
 midi = hz_to_midi(fconv), note = hz_to_note(fconv; octave=showoct, cents=showcents),
 back_to_hz = note_to_hz(hz_to_note(fconv)),
 roundtrip_slaney = mel_to_hz(hz_to_mel(fconv)))

# ╔═╡ a57299f3-f09f-4ff8-84d0-e6d4b5709ec7
let f = 1:10:8000
	plot(f, hz_to_mel.(f); label="Slaney (htk=false)", xguide="Hz", yguide="mel", title="mel scales")
	plot!(f, hz_to_mel.(f; htk=true); label="HTK (htk=true)")
	vline!([fconv]; label="$fconv Hz", ls=:dash, c=:gray)
end

# ╔═╡ 0ed7c07a-26fe-45d2-a6a0-476165e4e20b
md"""
Grids — bins: $(@bind ngrid Slider(8:4:64; default=24, show_value=true))
bins per octave: $(@bind bpo Select([12, 24, 36]))
tuning: $(@bind tuning Slider(-0.5:0.05:0.5; default=0, show_value=true))
"""

# ╔═╡ d1d4f574-4492-4dcb-b262-9660ff33071a
let
	p = plot(title="frequency grids ($ngrid points)", xguide="Hz", yticks=(1:4, ["fft", "mel slaney", "mel htk", "cqt"]),
	         legend=false, xscale=:log10, xlims=(20, sr / 2), ylims=(0.5, 4.5))
	g = [fft_frequencies(sr, 2(ngrid - 1))[2:end], mel_frequencies(ngrid; fmin=50, fmax=sr / 2),
	     mel_frequencies(ngrid; fmin=50, fmax=sr / 2, htk=true),
	     filter(<(sr / 2), cqt_frequencies(ngrid; fmin=55, bins_per_octave=bpo, tuning))]
	for (i, v) in enumerate(g)
		scatter!(p, v, fill(i, length(v)); ms=3)
	end
	p
end

# ╔═╡ 7ebec8d1-c9ca-4028-8607-b031cfa91208
let hop = 256
	(frame_10_first_sample = frames_to_samples(10, hop),
	 sample_5000_in_frame = samples_to_frames(5000, hop),
	 frame_10_time = frames_to_time(10, hop, sr),
	 time_1s_frame = time_to_frames(1.0, hop, sr),
	 sample_16001_time = samples_to_time(16001, sr),
	 time_1s_sample = time_to_samples(1.0, sr),
	 tempo_bpm_lags_1_to_4 = tempo_frequencies(Float64, 5, hop, sr)[2:end])
end

# ╔═╡ eaed5780-4f46-4848-9135-e1eac1ae617e
md"""
## 2. Decibels and loudness weighting

`power_to_db(S; ref, amin, top_db, min_db)` computes `10·log10(max(S, amin)/ref)`, `amplitude_to_db` the same with `20·log10`.

| parameter | effect |
|:--|:--|
| `ref` | 0 dB reference: a number, or a function of the data such as `maximum` (then the peak is 0 dB). |
| `amin` | floor applied before the log, so silence does not give `-Inf`. |
| `top_db` | clip everything more than `top_db` below the maximum: sets the visible dynamic range. `nothing` disables it. |
| `min_db` | absolute floor in dB. |

`db_to_power` and `db_to_amplitude` invert them. `A_weighting(f; min_db)` (and `B`, `C`, `D`) give the IEC loudness curves in dB; `perceptual_weighting(S, freq; weighting)` adds one to a power spectrogram in dB.
"""

# ╔═╡ 9cf732eb-086e-4bee-8124-b917bdce2938
wchoices = [A_weighting => "A", B_weighting => "B", C_weighting => "C", D_weighting => "D"];

# ╔═╡ a3aa1163-4633-4434-b986-c21f41b90688
md"""
dynamic range (top db): $(@bind topdb Slider(20:10:120; default=80, show_value=true))
weighting floor (min db): $(@bind wmin Slider(-120:10:-20; default=-80, show_value=true))
weighting on spectrogram: $(@bind wfun Select(wchoices))
"""

# ╔═╡ 2dd0b77f-d087-4cc5-ae02-179b02115e36
let f = exp10.(range(log10(10), log10(20000); length=400))
	plot(f, A_weighting.(f; min_db=wmin); label="A", xscale=:log10, xguide="Hz", yguide="dB", title="weighting curves (min_db = $wmin)")
	plot!(f, B_weighting.(f; min_db=wmin); label="B")
	plot!(f, C_weighting.(f; min_db=wmin); label="C")
	plot!(f, D_weighting.(f; min_db=wmin); label="D")
end

# ╔═╡ 996a36ac-cc07-488b-81ce-b1fe29446c10
begin
	stft = Stft(Frames(audio; winsize=512, winstep=128))
	S = get_spec(stft)
	fr_hz = collect(get_freq(stft))
	t_s = get_times(stft)
end;

# ╔═╡ 8212424c-ff6c-46e6-8793-f180625b58de
let
	D1 = power_to_db(S; ref=maximum, top_db=topdb)
	D2 = perceptual_weighting(S, fr_hz; weighting=wfun, ref=maximum, top_db=topdb)
	p1 = heatmap(t_s, fr_hz, D1; title="power_to_db, top_db=$topdb", xguide="s", yguide="Hz")
	p2 = heatmap(t_s, fr_hz, D2; title="perceptual_weighting ($(nameof(wfun)))", xguide="s")
	plot(p1, p2; layout=(1, 2), size=(720, 300))
end

# ╔═╡ 8ac52780-940c-4158-9f24-6309151c563c
md"""
## 3. Signal utilities

| function | parameters |
|:--|:--|
| `mu_compress(x; mu, quantize)` | μ-law companding. Larger `mu` expands quiet samples more; `quantize=true` returns integers in `-(mu+1)/2 : (mu-1)/2` (8-bit for `mu=255`). |
| `mu_expand(y; mu, quantize)` | the inverse; with quantisation the round trip is lossy, which you can hear below. |
| `normalize_signal(x; norm, dims)` | divide by the `norm`-norm (`Inf` peak, `1`, `2`), over everything or along `dims`. |
| `feature_scale(X; method, dims, corrected)` | audioFlux scalers per column (`dims=1`) or row: `:minmax`, `:standard`, `:maxabs`, `:robust`, `:center`, `:mean`, `:arctan`. |
| `temporal_db(x; base)` | peak and mean sample level in dB and the fraction of samples at or below `-base` dB. |
"""

# ╔═╡ 7864914d-828c-4c98-8e40-001f3972a56f
md"""
mu: $(@bind mu Select([3, 15, 63, 255, 1023]; default=255)) quantize $(@bind muq CheckBox(default=true))
"""

# ╔═╡ edf3cd3b-d8ca-4a82-8ee8-08eb331b0435
begin
	y_mu = mu_compress(x; mu, quantize=muq)
	x_mu = Float32.(mu_expand(y_mu; mu, quantize=muq))
	(levels_used = length(unique(y_mu)), max_error = maximum(abs, x_mu .- x),
	 snr_dB = 10log10(sum(abs2, x) / max(sum(abs2, x_mu .- x), eps())))
end

# ╔═╡ be416396-5774-4a03-8f88-82959fb4bc13
let u = range(-1, 1; length=401)
	p1 = plot(u, mu_compress(collect(u); mu, quantize=false); label="compress, mu=$mu", title="μ-law curve", xguide="input")
	p2 = plot(x[1:min(end, 800)]; label="original", title="first 800 samples")
	plot!(p2, x_mu[1:min(end, 800)]; label="after compress→expand", alpha=0.7)
	plot(p1, p2; layout=(1, 2))
end

# ╔═╡ d019ff0f-4836-44b9-99fe-d4bcdcbd9f8d
audio_player(x_mu, sr)

# ╔═╡ fd9a335b-2db9-4f36-a54b-e96a4437ff2a
md"""
norm: $(@bind nnorm Select([Inf => "Inf (peak)", 1 => "1", 2 => "2"]))
scaler: $(@bind smethod Select([:minmax, :standard, :maxabs, :robust, :center, :mean, :arctan]))
corrected std $(@bind scorr CheckBox(default=false))
temporal db base: $(@bind tbase Slider(6:3:36; default=18, show_value=true))
"""

# ╔═╡ 9610435d-93e3-4b4e-8db2-5262beab87d6
let
	xn = normalize_signal(x; norm=nnorm)
	# a small feature matrix: three spectral bands' energy per frame (frames × features)
	F = permutedims(vcat(sum(S[1:20, :], dims=1), sum(S[21:80, :], dims=1), sum(S[81:end, :], dims=1)))
	Fs = feature_scale(F; method=smethod, dims=1, corrected=scorr)
	p = plot(t_s, Fs; label=["low band" "mid band" "high band"], title="feature_scale(method=:$smethod)", xguide="s")
	mx, mn, quiet = temporal_db(x; base=tbase)
	md_ = (normalized_peak = maximum(abs, xn), normalized_l2 = sqrt(sum(abs2, xn)),
	       temporal_max_db = mx, temporal_mean_db = mn, quiet_fraction = quiet)
	[p, md_]
end

# ╔═╡ b7e33916-03f7-4773-b8aa-3c35cbeaa77c
md"""
### Synthesis

`tone(f; sr, duration, phi)`, `chirp(fmin, fmax; sr, duration, linear, phi)` (exponential sweep unless `linear=true`), `clicks(times; sr, click_freq, click_duration, length)` and `synth_f0(times, freqs, sr; amplitudes)` (a sinusoid following an interpolated pitch curve).
"""

# ╔═╡ 678c405d-8c68-4ecd-9960-e3d8b1e412d0
md"""
kind: $(@bind synkind Select(["tone", "chirp", "clicks", "synth_f0"]))
frequency / fmin: $(@bind synf Slider(50:10:2000; default=440, show_value=true))
fmax: $(@bind synf2 Slider(500:100:7000; default=4000, show_value=true))
linear chirp $(@bind synlin CheckBox(default=false))
phase φ: $(@bind synphi Slider(-3.14:0.01:3.14; default=-1.57, show_value=true))
click length (s): $(@bind clickdur Slider(0.01:0.01:0.2; default=0.05, show_value=true))
"""

# ╔═╡ d8c8ca91-657b-4840-8e2a-1765b2a3a8f2
syn = let d = 1.0
	if synkind == "tone"
		tone(synf; sr, duration=d, phi=synphi, T=Float32)
	elseif synkind == "chirp"
		chirp(synf, synf2; sr, duration=d, linear=synlin, phi=synphi, T=Float32)
	elseif synkind == "clicks"
		clicks(0.1:0.2:0.9; sr, click_freq=synf, click_duration=clickdur, length=sr, T=Float32)
	else
		Float32.(synth_f0([0.0, 0.3, 0.6, 1.0], [synf, synf2, synf, synf2], sr; amplitudes=[0.2, 1.0, 0.5, 1.0]))
	end
end;

# ╔═╡ 316fc9e5-23d2-422e-939b-6389d5fe2905
let st = Stft(Frames(syn, sr; winsize=512, winstep=128))
	plot(plot(range(0, 1; length=length(syn)), syn; label=synkind, xguide="s", legend=false, title=synkind),
	     plot(st; title="spectrogram"); layout=(1, 2), size=(720, 280))
end

# ╔═╡ 3aa38981-9d43-4beb-af1a-2355bf748000
audio_player(syn, sr)

# ╔═╡ 92af6a9e-231b-4e8f-bacf-83198990795a
md"""
### Silence trimming and splitting

`trim_silence(x; top_db, frame_length, hop_length)` drops leading and trailing frames quieter than `top_db` below the loudest frame; `split_silence` returns the intervals of every non-silent region.
Here the signal is padded with half a second of faint noise on each side.

| parameter | effect |
|:--|:--|
| `top_db` | threshold below the peak frame. Small values keep only the loudest parts (more splits), large values keep almost everything. |
| `frame_length` | RMS window. Longer windows bridge short gaps between words. |
| `hop_length` | resolution of the interval boundaries. |
"""

# ╔═╡ dd49d17a-a7ac-4124-9ecd-cf16030b6c73
md"""
threshold (top db): $(@bind sildb Slider(10:5:80; default=30, show_value=true))
frame length: $(@bind silfl Select([256, 512, 1024, 2048]; default=1024))
hop length: $(@bind silhop Select([64, 128, 256, 512]; default=256))
"""

# ╔═╡ 557a1dec-6678-443e-aca9-6a44a2ad55ef
let
	pad = 1f-3 .* randn(Xoshiro(1), Float32, sr ÷ 2)
	xs = vcat(pad, x, pad)
	kw = (; top_db=sildb, frame_length=silfl, hop_length=silhop)
	y, (a, b) = trim_silence(xs; kw...)
	iv = split_silence(xs; kw...)
	t = (0:length(xs)-1) ./ sr
	p = plot(t, xs; label="padded signal", c=:gray, xguide="s", title="trim keeps $(round((b - a + 1) / sr; digits=2)) s, split finds $(length(iv)) regions")
	vspan!(p, [(a - 1) / sr, (b - 1) / sr]; alpha=0.15, c=:green, label="trim_silence")
	for (k, (i, j)) in enumerate(iv)
		plot!(p, t[i:j], xs[i:j]; c=2, label=k == 1 ? "split_silence" : "")
	end
	p
end

# ╔═╡ 67ffcc8e-6665-42cb-8cbc-a3a6ca9c747e
md"""
### Linear prediction

`lpc(x, order)` fits an all-pole model `1 / A(z)` by Burg's method. Its frequency response is a smooth envelope of the frame's spectrum: with a low `order` only the broad slope survives; around `2 + sr/1000` poles it follows the speech formants; with a high order it starts tracing individual harmonics.
"""

# ╔═╡ 87e86bb3-99a6-43df-9741-33b4caafc246
md"""
order: $(@bind lpcorder Slider(2:1:60; default=18, show_value=true))
frame at (fraction of signal): $(@bind lpcpos Slider(0.05:0.05:0.95; default=0.4, show_value=true))
"""

# ╔═╡ 2658d03f-8eac-42c0-8ca7-13d6e546487c
let n = 512
	s = clamp(round(Int, lpcpos * length(x)), 1, length(x) - n)
	seg = x[s:s+n-1] .* hanning(n)
	a = lpc(Float64.(seg), lpcorder)
	nfft = 2048
	X = abs.(Audio911.FFTW.rfft(vcat(seg, zeros(nfft - n))))
	A = abs.(Audio911.FFTW.rfft(vcat(a, zeros(nfft - length(a)))))
	g = sqrt(sum(abs2, X) / sum(abs2, 1 ./ A))  # match energies for display
	f = fft_frequencies(sr, nfft)
	plot(f, 20log10.(X .+ 1e-9); label="frame spectrum", alpha=0.6, xguide="Hz", yguide="dB", title="LPC envelope, order $lpcorder")
	plot!(f, 20log10.(g ./ A); label="1/A(z)", lw=2)
end

# ╔═╡ 360e9429-da27-4dd4-a911-10a0715fe6ed
# `get_samplerate` reads only the header, without decoding the audio
[f => get_samplerate(joinpath(SAMPLES_DIR, f)) for f in ("test.wav", "test.flac", "test.ogg", "test.mp3")]

# ╔═╡ ba7c0735-7393-441f-b37a-7fe0e4c3f99e
md"""
## 4. DSP building blocks

| function | parameters |
|:--|:--|
| `hilbert(x)` | analytic signal; `abs` is the amplitude envelope, `angle` the instantaneous phase. |
| `czt(x, band; m)` | zoom FFT: `m` points between `band[1]` and `band[2]` (cycles per sample, multiply by `sr` for Hz). More points on a narrow band show detail an FFT of the same length cannot. `czt(x, m, w, a)` is the general chirp-Z form. |
| `xcorr(x, y; normalize)` | cross-correlation over lags `-(N-1):N-1`; `normalize=true` scales the autocorrelation to 1 at lag 0. |
| `convolve(a, b; mode)` | `:full` (all samples), `:same` (length of `a`, centred), `:valid` (no zero padding). |
"""

# ╔═╡ 55310a88-4600-4bcc-802d-1345e50d82a9
let n = min(length(x), sr ÷ 2)
	seg = x[1:n]
	env = abs.(hilbert(seg))
	t = (0:n-1) ./ sr
	plot(t, seg; label="signal", alpha=0.5, xguide="s", title="Hilbert envelope (first 0.5 s)")
	plot!(t, env; label="|hilbert(x)|", lw=2)
end

# ╔═╡ 25432c83-38ad-43c3-84d1-91fd1b48a85d
md"""
Zoom band centre: $(@bind cztc Slider(100:10:4000; default=455, show_value=true)) Hz
width: $(@bind cztw Slider(20:10:1000; default=100, show_value=true)) Hz
points m: $(@bind cztm Slider(32:32:1024; default=256, show_value=true))
"""

# ╔═╡ 8ea3b6e6-18bf-4a13-9a4e-d7bb7371b0d3
let n = 2048
	seg = Float64.(x[1:n]) .* hanning(n)
	lo, hi = max(cztc - cztw / 2, 0), min(cztc + cztw / 2, sr / 2)
	Z = czt(seg, (lo / sr, hi / sr); m=cztm)
	fz = range(lo, hi; length=cztm + 1)[1:end-1]
	F = abs.(Audio911.FFTW.rfft(seg)); ff = fft_frequencies(sr, n)
	keep = findall(f -> lo ≤ f ≤ hi, ff)
	plot(fz, abs.(Z); label="czt ($cztm points)", xguide="Hz", title="zoom FFT of the first $n samples (tones signal: 440 + 470 Hz)")
	scatter!(ff[keep], F[keep]; label="plain FFT bins", ms=4)
end

# ╔═╡ e27a89c7-1402-40ce-bb62-b2ace59724d6
md"""
Delay of a copy: $(@bind lagms Slider(0:1:200; default=40, show_value=true)) ms
noise added: $(@bind lagnoise Slider(0:0.05:1; default=0.2, show_value=true))
normalize $(@bind xcnorm CheckBox(default=true))
"""

# ╔═╡ 3a6f6732-22d6-492e-9589-a68ee9759396
let n = min(length(x), sr)
	d = round(Int, lagms * sr / 1000)
	a = x[1:n]
	b = vcat(zeros(Float32, d), a)[1:n] .+ lagnoise .* randn(Xoshiro(2), Float32, n)
	r = xcorr(b, a; normalize=xcnorm)
	lags = -(n - 1):(n - 1)
	win = abs.(lags) .≤ sr ÷ 4
	est = lags[argmax(r)]
	plot(1000 .* lags[win] ./ sr, r[win]; label="xcorr(delayed, original)", xguide="lag (ms)",
	     title="estimated delay = $(round(1000est / sr; digits=1)) ms (true $lagms ms)")
end

# ╔═╡ 3f178f0b-cd66-45b0-a29f-c02ced0e4dba
md"""
Smoothing kernel length: $(@bind klen Slider(1:2:401; default=81, show_value=true)) mode: $(@bind cmode Select([:same, :full, :valid]))
"""

# ╔═╡ d19827ea-c38b-41ec-8de9-d6336caa25d1
let
	env = abs.(x)
	k = hanning(klen); k ./= sum(k)
	y = convolve(env, k; mode=cmode)
	plot(env[1:min(end, sr)]; label="|x|", alpha=0.4, title="convolve(|x|, hanning($klen); mode=:$cmode): $(length(y)) samples from $(length(env))")
	plot!(y[1:min(end, sr)]; label="smoothed", lw=2)
end

# ╔═╡ 813e671f-18ff-4109-9c2a-f46ca8275e45
md"""
## 5. Time stretching and pitch shifting

`time_stretch(x, rate; winsize, winstep, window)` changes the duration by `1/rate` without changing the pitch: the complex STFT goes through `phase_vocoder(C, rate; hop)` and back through an overlap-add inverse STFT.
`pitch_shift(x, n_steps; winsize, winstep, window, quality)` stretches by `2^(-n_steps/12)` and resamples back to the original length.

| parameter | effect |
|:--|:--|
| `rate` | `>1` faster/shorter, `<1` slower/longer. |
| `n_steps` | semitones; fractional values are allowed. |
| `winsize` | long windows (4096) keep tonal music clean but smear transients ("phasiness"); short windows keep speech and drums crisp but roughen pitch. |
| `winstep` | hop; `winsize ÷ 4` (75 % overlap) is the usual choice, larger hops add artefacts. |
| `window` | analysis/synthesis window. |
| `quality` | resampler of `pitch_shift`: `:fast`, `:mid` or `:best` band-limited sinc. |
"""

# ╔═╡ edc86e3d-33fc-412d-9fd2-4b1f1c875b74
md"""
rate: $(@bind rate Slider(0.5:0.05:2.0; default=0.75, show_value=true))
semitones: $(@bind nsteps Slider(-12:0.5:12; default=4, show_value=true))
winsize: $(@bind tswin Select([512, 1024, 2048, 4096]; default=2048))
hop: $(@bind tshopdiv Select([2 => "winsize/2", 4 => "winsize/4", 8 => "winsize/8"]; default=4))
window: $(@bind tswindow Select([hanning => "hanning", hamming => "hamming", blackman => "blackman"]))
quality: $(@bind tsq Select([:fast, :mid, :best]))
"""

# ╔═╡ b6d7ccd3-d72c-442c-99b0-aa4c83089a63
begin
	stretched = time_stretch(x, rate; winsize=tswin, winstep=tswin ÷ tshopdiv, window=tswindow)
	shifted = pitch_shift(x, nsteps; winsize=tswin, winstep=tswin ÷ tshopdiv, window=tswindow, quality=tsq)
	(original_s = length(x) / sr, stretched_s = length(stretched) / sr, shifted_s = length(shifted) / sr)
end

# ╔═╡ 425d3e16-e441-45ab-9da3-087e3fcad9ad
md"""
original $(audio_player(x, sr))
time-stretched (rate $rate) $(audio_player(stretched, sr))
pitch-shifted ($nsteps semitones) $(audio_player(shifted, sr))
"""

# ╔═╡ e21f919b-bcf2-4a13-a578-d459f0b0b36d
let fr(v) = Stft(Frames(Float32.(v), sr; winsize=512, winstep=128))
	plot(plot(fr(x); title="original"), plot(fr(stretched); title="time_stretch"),
	     plot(fr(shifted); title="pitch_shift"); layout=(1, 3), size=(720, 280), colorbar=false)
end

# ╔═╡ 3551b824-08f5-4dad-a7f4-122662b16649
md"""
`phase_vocoder` can also be applied directly to any complex STFT, for instance to inspect how frames are resampled:
"""

# ╔═╡ ac883149-776e-4023-9695-fdbfd9741413
let C = get_complex(Stft(Frames(x, sr; winsize=1024, winstep=256)))
	D = phase_vocoder(C, rate; hop=256)
	(input_frames = size(C, 2), output_frames = size(D, 2), expected = ceil(Int, size(C, 2) / rate))
end

# ╔═╡ bc843258-c8df-4748-b616-68e5fba34d4c
md"""
## 6. Non-negative matrix factorisation

`nmf(V, k; divergence, max_iter, thresh, norm, init)` factors a non-negative spectrogram `V ≈ W·H`: the `k` columns of `W` are spectral templates, the rows of `H` their activations over time.
On the chord signal, two or three components separate the notes; on the tones-and-clicks signal one component captures the clicks.

| parameter | effect |
|:--|:--|
| `k` | number of components (rank). Too few merge sources; too many split one source into several. |
| `divergence` | `:kl` (Kullback-Leibler, good default for magnitude spectra), `:is` (Itakura-Saito, scale-invariant: quiet parts matter as much as loud ones), `:euclidean` (dominated by loud bins). |
| `max_iter`, `thresh` | stopping rule. |
| `norm` | normalisation of `W`'s columns: `:max`, `:l1` or `:l2`. |
| `init` | `:nndsvd` (deterministic SVD-based), `:audioflux` (audioFlux's ramps), or `(W0, H0)`. |
"""

# ╔═╡ 747837fe-61a7-4c62-8eb8-b5d6d3017cb8
md"""
k: $(@bind nmfk Slider(1:1:8; default=3, show_value=true))
divergence: $(@bind nmfdiv Select([:kl, :is, :euclidean]))
iterations: $(@bind nmfiter Select([10, 50, 100, 300]; default=100))
thresh: $(@bind nmfth Select([1e-2, 1e-3, 1e-4]; default=1e-3))
norm: $(@bind nmfnorm Select([:max, :l1, :l2]))
init: $(@bind nmfinit Select([:nndsvd, :audioflux]))
"""

# ╔═╡ 57c66a33-4b4b-4acd-a16f-cbc2d643cfb1
begin
	mstft = Stft(Frames(audio; winsize=1024, winstep=256); spectrum=magnitude)
	Vm = get_spec(mstft)[1:257, :]   # up to sr/4 keeps the demo fast
	W, H = nmf(Vm, nmfk; divergence=nmfdiv, max_iter=nmfiter, thresh=nmfth, norm=nmfnorm, init=nmfinit)
	(relative_error = sqrt(sum(abs2, Vm .- W * H) / sum(abs2, Vm)),)
end

# ╔═╡ c82e9d68-3dd2-405d-86a8-22f2fda82ae1
let f = collect(get_freq(mstft))[1:257], t = get_times(mstft)
	p1 = plot(f, W; xguide="Hz", title="templates W", label=permutedims(["c$i" for i in 1:nmfk]))
	p2 = plot(t, permutedims(H); xguide="s", title="activations H", legend=false)
	p3 = heatmap(t, f, log10.(Vm .+ 1e-6); title="V (log)", colorbar=false)
	p4 = heatmap(t, f, log10.(W * H .+ 1e-6); title="W·H (log)", colorbar=false)
	plot(p1, p2, p3, p4; layout=(2, 2), size=(720, 520))
end

# ╔═╡ 72ee695b-6af7-4725-b79b-30e2da98d4b8
md"""
Listen to one component, rebuilt by a soft mask `(Wᵢ Hᵢ) / (W H)` on the complex STFT: $(@bind nmfcomp Slider(1:nmfk; default=1, show_value=true))
"""

# ╔═╡ d2886bda-4949-4eca-b38a-76953b65a10d
let st = Stft(Frames(audio; winsize=1024, winstep=256); spectrum=magnitude)
	C = get_complex(st)
	M = zeros(Float32, size(C))
	M[1:257, :] .= (W[:, nmfcomp] * H[nmfcomp:nmfcomp, :]) ./ (W * H .+ 1f-9)
	y = istft(C .* M, 1024, 256; window=hanning, length=length(x))
	audio_player(y, sr)
end

# ╔═╡ a6b0ba8b-51db-4e5e-81a9-ebf325e58f3a
md"""
## 7. Hidden Markov models

A discrete `Hmm(p_init, A, B)` has an initial distribution, an `S × S` transition matrix `A` and an `S × K` emission matrix `B`.

| function | meaning |
|:--|:--|
| `hmm_generate(h, n; rng)` | sample `n` states and observations. |
| `hmm_predict(h, obs)` | log-likelihood `log P(obs | model)`. |
| `hmm_decode(h, obs)` | most likely state path (Viterbi) and its log probability. |
| `hmm_train(h, obs; max_iter, tol)` | Baum-Welch re-estimation from a starting model. |
| `viterbi(prob, A; p_init)` | Viterbi on any `states × frames` likelihood matrix — for example frame-wise class scores from a feature. |

Toy model: two states, *silence* and *speech*, emitting a quantised loudness symbol (1 quiet, 2 medium, 3 loud).
`stay` is the probability of remaining in the same state: high values make the decoded path sticky, which is how Viterbi smoothing removes flicker.
"""

# ╔═╡ 8dd64577-b456-47d7-b543-51fddbe28b8f
md"""
stay probability: $(@bind stay Slider(0.5:0.01:0.99; default=0.9, show_value=true))
emission sharpness: $(@bind sharp Slider(0.4:0.05:0.9; default=0.7, show_value=true))
generated length: $(@bind hmmn Slider(50:50:1000; default=300, show_value=true))
seed: $(@bind hmmseed Slider(1:20; default=1, show_value=true))
Baum-Welch iterations: $(@bind bwiter Slider(1:1:200; default=50, show_value=true))
"""

# ╔═╡ a89b56de-4b91-4f70-aeda-8c00c0ed3721
begin
	rest = (1 - sharp) / 2
	h_true = Hmm([0.5, 0.5], [stay 1-stay; 1-stay stay], [sharp 1-sharp-rest rest; rest 1-sharp-rest sharp])
	states, obs = hmm_generate(h_true, hmmn; rng=Xoshiro(hmmseed))
	h0 = Hmm([0.5, 0.5], [0.6 0.4; 0.4 0.6], [0.4 0.35 0.25; 0.25 0.35 0.4])
	h_fit = hmm_train(h0, obs; max_iter=bwiter, tol=1e-4)
	path, logp = hmm_decode(h_fit, obs)
	(loglik_true = hmm_predict(h_true, obs), loglik_start = hmm_predict(h0, obs),
	 loglik_trained = hmm_predict(h_fit, obs), viterbi_logp = logp,
	 decode_accuracy = count(path .== states) / hmmn)
end

# ╔═╡ 0b0ac166-d0e6-4e83-8a06-3184615cc925
plot(plot(h_true; title=["true A" "true B"]), plot(h_fit); layout=(2, 1), size=(720, 480))

# ╔═╡ 47fe0b6a-f587-42d9-a653-c69bfc416b6b
let n = min(hmmn, 200)
	plot(obs[1:n]; seriestype=:steppost, label="observed symbol", alpha=0.5, title="first $n steps", xguide="step")
	plot!(states[1:n] .+ 0.05; seriestype=:steppost, label="true state", lw=2)
	plot!(path[1:n] .- 0.05; seriestype=:steppost, label="Viterbi path", lw=2, ls=:dash)
end

# ╔═╡ 44922ff3-0687-4961-a0d3-fc50bc80e2ef
md"""
### Viterbi on real audio: voice activity

`viterbi` also takes a continuous likelihood matrix. Here frame loudness (dB) is turned into a soft *speech* probability with a logistic curve around a threshold, and Viterbi with the `stay` probability above decides speech/silence per frame.
"""

# ╔═╡ 644a6b1a-514f-4813-b546-49c656c06c96
md"""
threshold below peak: $(@bind vadth Slider(-60:1:-5; default=-30, show_value=true)) dB
"""

# ╔═╡ 674b5dbd-e94a-44c6-ba31-ce04704d6fe7
let
	e = vec(sum(S, dims=1))
	edb = power_to_db(e; ref=maximum, top_db=nothing)
	p_speech = 1 ./ (1 .+ exp.(-(edb .- vadth) ./ 3))
	prob = permutedims(hcat(1 .- p_speech, p_speech))
	vpath, _ = viterbi(prob, [stay 1-stay; 1-stay stay])
	p = plot(t_s, edb; label="frame level (dB)", xguide="s", title="Viterbi VAD (stay = $stay)", legend=:bottomright)
	hline!(p, [vadth]; label="threshold", ls=:dash)
	plot!(twinx(p), t_s, [p_speech .> 0.5, vpath .- 1]; seriestype=:steppost, label=["raw threshold" "viterbi"], ylims=(-0.1, 1.1), legend=:topright)
end

# ╔═╡ 355b8d91-7b0a-4ee8-b4b2-7e72788d5364
md"""
## Next

This is the last notebook of the tour. Go back to the index in `notebooks/README.md`:

1. 01_loading_and_frames · 2. 02_stft_and_filterbanks · 3. 03_cepstra · 4. 04_spectral_descriptors · 5. 05_time_frequency · 6. 06_discrete_wavelets_and_decompositions · 7. 07_music_and_rhythm · 8. 08_time_domain_and_pitch · 9. 09_signal_processing_and_classic
"""

# ╔═╡ Cell order:
# ╟─8bfb5dcf-f51b-4493-b6d3-712211a73619
# ╟─03e3cc2f-e47f-40e1-8ffc-fdfadedaadef
# ╟─aeae8f57-ea68-4e68-aa45-b296168bb48a
# ╟─898ec42b-2193-4982-8e48-c7094b95783a
# ╠═613a1703-9714-4546-ad94-2089a6e8a842
# ╟─cca58518-0890-4837-8370-20b163f96b22
# ╟─5914d8d3-fb68-4061-8cd8-3369fbd7937f
# ╠═70e2424e-29a0-4e93-a491-31ad38d54cd3
# ╠═a57299f3-f09f-4ff8-84d0-e6d4b5709ec7
# ╟─0ed7c07a-26fe-45d2-a6a0-476165e4e20b
# ╠═d1d4f574-4492-4dcb-b262-9660ff33071a
# ╠═7ebec8d1-c9ca-4028-8607-b031cfa91208
# ╟─eaed5780-4f46-4848-9135-e1eac1ae617e
# ╟─9cf732eb-086e-4bee-8124-b917bdce2938
# ╟─a3aa1163-4633-4434-b986-c21f41b90688
# ╠═2dd0b77f-d087-4cc5-ae02-179b02115e36
# ╠═996a36ac-cc07-488b-81ce-b1fe29446c10
# ╠═8212424c-ff6c-46e6-8793-f180625b58de
# ╟─8ac52780-940c-4158-9f24-6309151c563c
# ╟─7864914d-828c-4c98-8e40-001f3972a56f
# ╠═edf3cd3b-d8ca-4a82-8ee8-08eb331b0435
# ╠═be416396-5774-4a03-8f88-82959fb4bc13
# ╠═d019ff0f-4836-44b9-99fe-d4bcdcbd9f8d
# ╟─fd9a335b-2db9-4f36-a54b-e96a4437ff2a
# ╠═9610435d-93e3-4b4e-8db2-5262beab87d6
# ╟─b7e33916-03f7-4773-b8aa-3c35cbeaa77c
# ╟─678c405d-8c68-4ecd-9960-e3d8b1e412d0
# ╠═d8c8ca91-657b-4840-8e2a-1765b2a3a8f2
# ╠═316fc9e5-23d2-422e-939b-6389d5fe2905
# ╠═3aa38981-9d43-4beb-af1a-2355bf748000
# ╟─92af6a9e-231b-4e8f-bacf-83198990795a
# ╟─dd49d17a-a7ac-4124-9ecd-cf16030b6c73
# ╠═557a1dec-6678-443e-aca9-6a44a2ad55ef
# ╟─67ffcc8e-6665-42cb-8cbc-a3a6ca9c747e
# ╟─87e86bb3-99a6-43df-9741-33b4caafc246
# ╠═2658d03f-8eac-42c0-8ca7-13d6e546487c
# ╠═360e9429-da27-4dd4-a911-10a0715fe6ed
# ╟─ba7c0735-7393-441f-b37a-7fe0e4c3f99e
# ╠═55310a88-4600-4bcc-802d-1345e50d82a9
# ╟─25432c83-38ad-43c3-84d1-91fd1b48a85d
# ╠═8ea3b6e6-18bf-4a13-9a4e-d7bb7371b0d3
# ╟─e27a89c7-1402-40ce-bb62-b2ace59724d6
# ╠═3a6f6732-22d6-492e-9589-a68ee9759396
# ╟─3f178f0b-cd66-45b0-a29f-c02ced0e4dba
# ╠═d19827ea-c38b-41ec-8de9-d6336caa25d1
# ╟─813e671f-18ff-4109-9c2a-f46ca8275e45
# ╟─edc86e3d-33fc-412d-9fd2-4b1f1c875b74
# ╠═b6d7ccd3-d72c-442c-99b0-aa4c83089a63
# ╠═425d3e16-e441-45ab-9da3-087e3fcad9ad
# ╠═e21f919b-bcf2-4a13-a578-d459f0b0b36d
# ╟─3551b824-08f5-4dad-a7f4-122662b16649
# ╠═ac883149-776e-4023-9695-fdbfd9741413
# ╟─bc843258-c8df-4748-b616-68e5fba34d4c
# ╟─747837fe-61a7-4c62-8eb8-b5d6d3017cb8
# ╠═57c66a33-4b4b-4acd-a16f-cbc2d643cfb1
# ╠═c82e9d68-3dd2-405d-86a8-22f2fda82ae1
# ╟─72ee695b-6af7-4725-b79b-30e2da98d4b8
# ╠═d2886bda-4949-4eca-b38a-76953b65a10d
# ╟─a6b0ba8b-51db-4e5e-81a9-ebf325e58f3a
# ╟─8dd64577-b456-47d7-b543-51fddbe28b8f
# ╠═a89b56de-4b91-4f70-aeda-8c00c0ed3721
# ╠═0b0ac166-d0e6-4e83-8a06-3184615cc925
# ╠═47fe0b6a-f587-42d9-a653-c69bfc416b6b
# ╟─44922ff3-0687-4961-a0d3-fc50bc80e2ef
# ╟─644a6b1a-514f-4813-b546-49c656c06c96
# ╠═674b5dbd-e94a-44c6-ba31-ce04704d6fe7
# ╟─355b8d91-7b0a-4ee8-b4b2-7e72788d5364
