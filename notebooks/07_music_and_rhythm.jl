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

# ╔═╡ cabcc17f-1910-48f2-a303-5c5bf6dd4957
begin
	import Pkg
	# the notebook environment next to this file pins Audio911 (from this repository), PlutoUI and Plots
	Pkg.activate(@__DIR__)
	Pkg.instantiate()
	using Audio911, PlutoUI, Plots
	using Statistics: mean, median
	include(joinpath(@__DIR__, "common.jl"))
	gr(); default(size=(720, 300), legend=:topright, titlefontsize=10, guidefontsize=9)
end

# ╔═╡ aa8f59a3-9171-4abf-b07b-14a9410925ea
md"""
# Audio911 · 7 — Music and rhythm

Features that describe **harmony** (chroma, tonnetz), **rhythm** (onsets, tempo, beats) and the **structure** of a spectrogram (harmonic/percussive separation, gating, PCEN, harmonic count).
They all consume a front end built exactly as in notebooks 1–2, so any `Stft`, `Cqt` or filterbank output can feed them.

Move a slider or change a menu and every cell that depends on it re-runs.
"""

# ╔═╡ 4347f193-f7cf-45ee-9ab9-c90f71d24430
TableOfContents()

# ╔═╡ d59143f4-8442-489b-b4a0-af182c90f822
md"""
## 0. The signal

Besides the shared menu, this notebook adds a **click track**: clicks at a chosen tempo over a low drone, so tempo and beat estimates can be checked against the truth.
The chord signal (C major, then A minor) is the natural choice for chroma and tonnetz; the tones-and-clicks signal for onsets and HPSS.

Signal: $(@bind signame Select([SIGNALS; "rhythm" => "Click track over a 220 Hz drone"]; default="chord"))
click-track tempo (BPM): $(@bind truebpm Slider(60:5:180; default=120, show_value=true))
"""

# ╔═╡ e27e48d5-9ed0-47df-9442-0530d878c91b
begin
	sr = 16000
	function rhythm_signal(bpm; sr=16000, duration=6.0)
		n = round(Int, duration * sr)
		c = clicks(0.25:60/bpm:duration-0.2; sr, click_freq=1500, click_duration=0.05, length=n, T=Float32)
		x = c .+ 0.2f0 .* tone(220; sr, duration, T=Float32)
		AudioFile(Float32.(x ./ maximum(abs, x)), sr)
	end
	audio = signame == "rhythm" ? rhythm_signal(truebpm; sr) : demo_signal(signame; sr)
	x = vec(get_data(audio))
	audio_player(audio)
end

# ╔═╡ b462f592-e9c7-4285-910e-7486228acb3c
md"""
One STFT is shared by most sections below.
Harmony wants long frames (fine frequency resolution), rhythm wants a short hop (fine time resolution); 1024 samples with a 256 hop is a reasonable compromise at 16 kHz.
`keep_complex=true` keeps the phase, needed by the phase-based novelties and by the HPSS signal reconstruction.

winsize: $(@bind winsize Select([512, 1024, 2048, 4096]; default=1024))
hop: $(@bind hop Select([128, 256, 512]; default=256))
"""

# ╔═╡ 12e53ef9-b230-42f6-80fa-a47d3a3551a1
stft = Stft(Frames(audio; winsize, winstep=hop); keep_complex=true)

# ╔═╡ d2bb3759-1c17-4911-b03d-cc45ce38377d
md"""
## 1. Chroma from the STFT

`Chroma(spec; norm, kwargs...)` folds the spectrum onto the pitch classes with a `chroma_fbank` (librosa `filters.chroma`).
Every STFT bin is mapped to a fractional pitch class `nchroma · hz_to_octs(f)` and spread with a Gaussian one bin wide.

| parameter | effect |
|:--|:--|
| `nchroma` | number of pitch classes. 12 is the semitone grid; 24 or 36 resolve quarter or third tones, useful for out-of-tune material. |
| `tuning` | offset of the reference A440 in fractions of a bin (−0.5…0.5). A recording tuned 30 cents flat needs `tuning ≈ -0.3` to land on the right classes. |
| `ctroct` | centre (in octaves above A440/16 ≈ 27.5 Hz) of the Gaussian octave weighting; 5 ≈ 880 Hz. |
| `octwidth` | width in octaves of that weighting. Smaller values focus the chroma on a register; `nothing` weights every octave equally, so low rumble and high harmonics count as much as the melody. |
| `base_c` | `true` puts C in row 1; `false` starts at A. |
| `norm` | per-frame normalisation: `Inf` (max = 1, librosa's default), `1`, `2`, or `nothing` to keep absolute energy (silence stays dark). |
"""

# ╔═╡ 0d08feb5-5c62-4487-9cab-c2c80a909075
md"""
nchroma: $(@bind nch Select([12, 24, 36]))
tuning: $(@bind tun Slider(-0.5:0.05:0.5; default=0, show_value=true))
ctroct: $(@bind ctr Slider(2.0:0.5:8.0; default=5.0, show_value=true))

octave weighting: $(@bind useoct CheckBox(default=true))
octwidth: $(@bind octw Slider(0.5:0.25:4; default=2, show_value=true))
`base_c` $(@bind basec CheckBox(default=true))
norm: $(@bind chromanorm Select([Inf => "Inf (max)", 1 => "1", 2 => "2", nothing => "none"]))
"""

# ╔═╡ 09b76ba9-c937-4655-b9ff-f26650dd42b3
begin
	cfb = chroma_fbank(stft; nchroma=nch, tuning=tun, ctroct=ctr, octwidth=useoct ? octw : nothing, base_c=basec)
	chromagram = Chroma(stft, cfb; norm=chromanorm)
end

# ╔═╡ 1c85d684-cf0a-4980-a556-f16667d02c16
let
	p1 = plot(cfb; xlims=(0, 4000), title="chroma filterbank (first 4 kHz)")
	p2 = plot(chromagram; title="chroma of the STFT")
	plot(p1, p2; layout=(2, 1), size=(720, 520))
end

# ╔═╡ a3250ae7-0fe5-476d-ac6d-7ae50f0e2f76
md"""
With the chord signal the first half lights up C, E and G and the second half A, C and E; the shared C and E rows stay lit across the change.
With 24 classes each note occupies every second row; changing `tuning` by ±0.5 moves the energy between rows.

`hz_to_octs(f; tuning, bins_per_octave)` is the octave scale used by the filterbank: A4 = 440 Hz sits at 4 octaves above 27.5 Hz.
"""

# ╔═╡ ef361cbe-b5eb-45a7-9132-8b764dcae0c7
[(note = n, Hz = round(note_to_hz(n); digits=2), octs = round(hz_to_octs(note_to_hz(n); tuning=tun); digits=3)) for n in ("A0", "C4", "A4", "E5")]

# ╔═╡ 3d6770c8-f9a1-4b71-b9da-8215af26fbd4
md"""
## 2. Chroma from the constant-Q transform

A CQT already has a logarithmic frequency axis, so its chroma is a plain folding: `cqt_chroma_fbank(cqt; nchroma)` assigns every CQT bin to exactly one pitch class (audioFlux `chroma_cqtFilterBank`), and `Chroma(cqt; nchroma, norm)` applies it.
Low notes are resolved much better than with an STFT of the same hop.

| parameter | effect |
|:--|:--|
| `fmin` | lowest CQT bin. Row 1 is always C, whatever `fmin` is. |
| `bins_per_octave` | CQT resolution; must be a multiple of `nchroma`. More bins per octave give sharper pitch classes but longer kernels (more smearing in time). |
| octaves | `nbins = octaves · bins_per_octave` bins are computed. |
| `nchroma` | pitch classes; `bins_per_octave ÷ nchroma` neighbouring bins are summed into each. |
"""

# ╔═╡ a3d5574a-901a-4ae8-8c4b-70df3ff518b9
md"""
fmin: $(@bind cqtfmin Select([32.70 => "C1 (32.7 Hz)", 65.41 => "C2 (65.4 Hz)", 130.81 => "C3 (130.8 Hz)"]; default=65.41))
bins per octave: $(@bind cqtbpo Select([12, 24, 36]; default=36))
octaves: $(@bind cqtoct Slider(3:6; default=5, show_value=true))
CQT nchroma: $(@bind cqtnchroma Select([12, 6, 4, 3]))
"""

# ╔═╡ b042f159-2337-4332-9e6b-3275a3524a46
begin
	cqt = Cqt(Frames(audio; winsize=512, winstep=hop, center=true);
	          fmin=cqtfmin, bins_per_octave=cqtbpo, nbins=cqtoct * cqtbpo)
	cqt_fb = cqt_chroma_fbank(cqt; nchroma=cqtbpo % cqtnchroma == 0 ? cqtnchroma : 12)
	cqt_chroma = Chroma(cqt, cqt_fb; norm=chromanorm)
end

# ╔═╡ 30e1a246-0f0d-4115-bb59-8a6162be9c18
let
	p1 = heatmap(get_freq(cqt), 0:size(get_data(cqt_fb), 1)-1, get_data(cqt_fb); xscale=:log10,
	             title="CQT → chroma folding (each bin belongs to one class)", xguide="Hz", yguide="class", c=:grays, colorbar=false)
	p2 = plot(cqt_chroma; title="chroma of the CQT")
	plot(p1, p2; layout=(2, 1), size=(720, 520))
end

# ╔═╡ 2d6f32f0-ee01-456b-b299-bcd6119cb2c6
md"""
## 3. Tonnetz

`Tonnetz(chroma)` (or `Tonnetz(spec; chroma kwargs...)`) projects each normalised chroma frame onto three circles of the *tonal network*: fifths (radius 1), minor thirds (radius 1) and major thirds (radius 0.5), each as an x/y pair.
Chords that share many notes land close together, so a harmonic change is a jump in these six curves.
It inherits every chroma parameter above (it uses the STFT chroma).
"""

# ╔═╡ 9a81cb41-f97e-441c-9b0b-296888b2bf67
let tz = Tonnetz(chromagram)
	plot(plot(tz), plot(get_times(tz), get_data(tz); label=["5th x" "5th y" "m3 x" "m3 y" "M3 x" "M3 y"], xguide="Time (s)", legend=:outerright);
	     layout=(2, 1), size=(720, 480))
end

# ╔═╡ 6a6aa3ac-fa31-4cbe-a9b3-b774bfe0cb6c
md"""
## 4. Onset strength

`OnsetStrength(spec; lag, max_size, detrend, aggregate)` is the spectral-flux envelope of Böck & Widmer (librosa `onset_strength`): the spectrogram in dB, the positive increase of every bin from frame `t - lag` to frame `t`, aggregated over bins.

| parameter | effect |
|:--|:--|
| input | librosa uses a mel spectrogram; a raw STFT gives more weight to the (many) high bins. |
| `lag` | compare with the frame `lag` hops back. Larger lags react to slower attacks and give broader peaks. |
| `max_size` | compare against a running maximum over `max_size` bins of the previous frame (SuperFlux). Values > 1 suppress vibrato, whose energy only moves to a neighbouring bin. |
| `detrend` | remove the slowly varying offset of the envelope with the `[1, -1] / [1, -0.99]` filter. |
| `aggregate` | how bins are combined: `mean` (librosa), `median` (robust to a few noisy bins), `maximum` (the single strongest increase). |
"""

# ╔═╡ cc4c6159-861e-48a5-a949-3a8811d2cbc3
md"""
input: $(@bind onsetin Select(["mel" => "MelSpec (64 bands)", "stft" => "Stft"]))
lag: $(@bind lagf Slider(1:6; default=1, show_value=true))
`max_size`: $(@bind maxsize Slider(1:2:9; default=1, show_value=true))
detrend $(@bind detr CheckBox(default=false))
aggregate: $(@bind agg Select([mean => "mean", median => "median", maximum => "maximum"]))
"""

# ╔═╡ 71dca424-9e48-447e-a5da-f3d5b26e10d2
env = OnsetStrength(onsetin == "mel" ? MelSpec(stft; nbands=64) : stft; lag=lagf, max_size=maxsize, detrend=detr, aggregate=agg)

# ╔═╡ 40e0d280-0f71-48da-8f7d-a28511231d84
md"""
### Picking the onsets

`onset_detect(env; delta, normalize, pre_max, post_max, pre_avg, post_avg, wait)` normalises the envelope and calls `peak_pick`: a frame is an onset when it is the local maximum of `[n-pre_max, n+post_max)`, exceeds the local mean over `[n-pre_avg, n+post_avg)` by `delta`, and comes more than `wait` frames after the previous onset.
The window defaults are librosa's (30 ms, 1 frame, 100 ms, 100 ms + 1 frame, 30 ms); a slider at 0 keeps the default.

| parameter | effect |
|:--|:--|
| `delta` | threshold above the local mean. Raise it to keep only strong attacks. |
| `normalize` | divide the envelope by its maximum first, so `delta` is relative. |
| `pre_max`, `post_max` | the local-maximum window. Wider windows keep only the strongest of nearby peaks. |
| `pre_avg`, `post_avg` | the local-mean window the threshold is measured against. |
| `wait` | minimum number of frames between onsets (refractory period). |
"""

# ╔═╡ 9d9bb236-c34e-48a2-ab83-96857013fa5f
md"""
delta: $(@bind deltaf Slider(0:0.01:0.5; default=0.07, show_value=true))
normalize $(@bind normz CheckBox(default=true))

`pre_max`: $(@bind premax Slider(0:20; default=0, show_value=true))
`post_max`: $(@bind postmax Slider(0:20; default=0, show_value=true))
`pre_avg`: $(@bind preavg Slider(0:40; default=0, show_value=true))
`post_avg`: $(@bind postavg Slider(0:40; default=0, show_value=true))
wait: $(@bind waitf Slider(0:40; default=0, show_value=true))
"""

# ╔═╡ f80eb0c3-95d0-4329-a17a-b40a8f24717a
begin
	dflt(v) = v == 0 ? nothing : v
	onsets = onset_detect(env; delta=deltaf, normalize=normz, pre_max=dflt(premax), post_max=dflt(postmax),
	                      pre_avg=dflt(preavg), post_avg=dflt(postavg), wait=dflt(waitf))
	onset_times = get_times(env)[onsets]
	(count = length(onsets), times_s = round.(onset_times; digits=3))
end

# ╔═╡ 7cc00c72-4547-49cf-a76d-87e28008e3cf
let t = get_times(env), tx = (0:length(x)-1) ./ sr
	p1 = plot(env; label="onset strength")
	scatter!(p1, t[onsets], get_data(env)[onsets]; label="onsets", ms=4)
	p2 = plot(tx, x; label="signal", alpha=0.6, xguide="Time (s)")
	vline!(p2, onset_times; label="onsets", lw=1.5)
	plot(p1, p2; layout=(2, 1), size=(720, 460))
end

# ╔═╡ a17e438c-7a4d-49b5-b5d3-08f01e0a7260
md"""
Listen to the signal with a click (at 3 kHz) on every detected onset:
"""

# ╔═╡ 2e4f03ea-b910-4c14-9c2d-563be9de8313
begin
	with_clicks(ts) = x .+ 0.5f0 .* clicks(ts; sr, click_freq=3000, click_duration=0.04, length=length(x), T=Float32)
	audio_player(with_clicks(onset_times), sr)
end

# ╔═╡ 5b56ee70-3f92-4b9f-9241-1c5b4b903164
md"""
`peak_pick` itself works on any vector with explicit windows in samples; `onset_detect` is exactly this call with the defaults resolved for the frame rate:
"""

# ╔═╡ 6f6d5c67-7efe-4091-b028-be110c4c6b36
let fps = sr / hop, e = get_data(env) ./ maximum(get_data(env))
	pk = peak_pick(e; pre_max=floor(Int, 0.03fps), post_max=1, pre_avg=floor(Int, 0.1fps),
	               post_avg=floor(Int, 0.1fps) + 1, delta=0.07, wait=floor(Int, 0.03fps))
	(peak_pick = pk, onset_detect_defaults = onset_detect(env))
end

# ╔═╡ d745f47d-4a3a-4073-bbe0-aed334e831f5
md"""
## 5. Novelty functions

`Novelty(spec; method, filter_order, kwargs...)` builds an onset envelope from any spectral descriptor (audioFlux's `Onset`), scaled to `[0, 1]`.
Magnitude methods respond to energy changes; the phase and complex-domain methods (`SpectralPd`, `SpectralWpd`, `SpectralNwpd`, `SpectralCd`, `SpectralRcd`) also detect soft onsets where only the phase changes, and need an STFT built with `keep_complex=true`.

| parameter | effect |
|:--|:--|
| `method` | the descriptor: flux (energy increase), HFC (high-frequency content, good for percussive clicks), SD / SF (spectral difference / its rectified version), MKL (log-ratio), broadband (count of rising bins), phase deviation variants, complex-domain variants. |
| `filter_order` | replace the spectrogram by its running maximum over this many bins first (magnitude methods only), like `max_size` above. |
| `kwargs` | forwarded to the descriptor (see notebook 4). |
"""

# ╔═╡ c341bbe7-7992-4d9f-b9af-96962e92e6b8
md"""
method: $(@bind novmethod Select([SpectralFlux => "SpectralFlux", SpectralHfc => "SpectralHfc", SpectralSd => "SpectralSd", SpectralSf => "SpectralSf", SpectralMkl => "SpectralMkl", SpectralBroadband => "SpectralBroadband", SpectralPd => "SpectralPd (phase)", SpectralWpd => "SpectralWpd (phase)", SpectralNwpd => "SpectralNwpd (phase)", SpectralCd => "SpectralCd (complex)", SpectralRcd => "SpectralRcd (complex)"]))
`filter_order`: $(@bind filtorder Slider(1:2:9; default=1, show_value=true))
"""

# ╔═╡ e0ba1427-e5a5-444d-81b6-db3b8293820b
let nov = Novelty(stft; method=novmethod, filter_order=filtorder)
	idx = onset_detect(nov; delta=deltaf)
	p = plot(nov; label=string(nameof(novmethod)))
	scatter!(p, get_times(nov)[idx], get_data(nov)[idx]; label="onsets ($(length(idx)))", ms=4)
	plot!(p, get_times(env), get_data(env) ./ maximum(get_data(env)); label="OnsetStrength (scaled)", alpha=0.5, ls=:dash)
end

# ╔═╡ 1f50005d-b68d-416c-b3bc-ac8757a65101
md"""
## 6. Tempogram and global tempo

`Tempogram(env; win_length, center)` is the local autocorrelation of the onset envelope around every frame (Grosche, Müller & Kurth; librosa `tempogram`): row `k` is a lag of `k-1` frames, i.e. a tempo of `60·sr/(hop·lag)` BPM (`tempo_frequencies`).
`tempo(env; start_bpm, std_bpm, ac_size, max_tempo, aggregate)` averages it over time and weights it with a log-normal prior around `start_bpm`.

| parameter | effect |
|:--|:--|
| `win_length` | autocorrelation window in frames. Longer windows give a steadier, sharper tempo estimate but follow tempo changes slowly. |
| `center` | pad so every column is centred on its frame. |
| `start_bpm` | centre of the tempo prior. Periodic signals have peaks at multiples of the true lag; the prior picks among them (tempo doubling/halving). |
| `std_bpm` | prior width in octaves. Small values force the estimate near `start_bpm`. |
| `ac_size` | autocorrelation window in seconds used by `tempo`. |
| `max_tempo` | ignore tempi above this. |
| `aggregate` | how the tempogram is summarised over time (`mean`, `median`, `maximum`). |
"""

# ╔═╡ 4808b9d0-d9f7-4202-80c4-d91213aaadb3
md"""
`win_length` (frames): $(@bind winlen Slider(32:32:512; default=384, show_value=true))
center $(@bind tgcenter CheckBox(default=true))

`start_bpm`: $(@bind startbpm Slider(40:5:240; default=120, show_value=true))
`std_bpm`: $(@bind stdbpm Slider(0.1:0.1:3; default=1, show_value=true))
`ac_size` (s): $(@bind acsize Slider(1:0.5:10; default=8, show_value=true))
`max_tempo`: $(@bind maxtempo Slider(120:20:400; default=320, show_value=true))
tempo aggregate: $(@bind tempoagg Select([mean => "mean", median => "median", maximum => "maximum"]))
"""

# ╔═╡ 3e4d98f5-05a4-4352-8655-d5acc0264865
begin
	tg = Tempogram(env; win_length=winlen, center=tgcenter)
	bpm_est = tempo(env; start_bpm=startbpm, std_bpm=stdbpm, ac_size=acsize, max_tempo=maxtempo, aggregate=tempoagg)
	(estimated_bpm = round(bpm_est; digits=1), truebpm = signame == "rhythm" ? truebpm : missing)
end

# ╔═╡ a8e49518-9c89-48b9-9709-59dc42c9aed5
let bpms = get_freq(tg), agg = vec(mean(get_spec(tg); dims=2))
	lag_est = 60 * sr / (hop * bpm_est)
	p1 = plot(tg; title="tempogram (row = lag in frames)")
	hline!(p1, [lag_est]; label="estimated tempo lag", c=:white, ls=:dash)
	k = 2:length(bpms)
	p2 = plot(bpms[k], agg[k]; xscale=:log2, xlims=(30, 480), xguide="BPM", label="mean autocorrelation")
	vline!(p2, [bpm_est]; label="tempo = $(round(bpm_est; digits=1))")
	plot(p1, p2; layout=(2, 1), size=(720, 520))
end

# ╔═╡ 1bbaf16a-24a0-4a11-8304-25b3e391c660
md"""
## 7. Beat tracking

`beat_track(env; bpm, tightness, trim)` (Ellis 2007; librosa `beat_track`) scores the envelope against a Gaussian comb of the tempo period and finds the best beat sequence by dynamic programming.
It returns `(bpm, beat_frames)`.

| parameter | effect |
|:--|:--|
| `bpm` | tempo to track; by default the `tempo` estimate. Forcing half or double the true tempo tracks every other beat or adds off-beats. |
| `tightness` | penalty for deviating from the period. High values give an even grid even through syncopation; low values let beats follow the onsets. |
| `trim` | drop weak beats at the start and end. |
"""

# ╔═╡ 76738ad4-3593-4f8d-8745-e50663e70b18
md"""
use estimated tempo $(@bind autobpm CheckBox(default=true))
forced bpm: $(@bind forcedbpm Slider(40:5:240; default=120, show_value=true))
tightness: $(@bind tight Slider(1:10:401; default=101, show_value=true))
trim $(@bind trimb CheckBox(default=true))
"""

# ╔═╡ e8be2ff5-c00c-4959-8863-667d29a86c4d
begin
	bt_bpm, beats = beat_track(env; bpm=autobpm ? nothing : forcedbpm, tightness=tight, trim=trimb)
	beat_times = get_times(env)[beats]
	let p = plot((0:length(x)-1) ./ sr, x; label="signal", alpha=0.5, xguide="Time (s)", title="beats at $(round(bt_bpm; digits=1)) BPM")
		vline!(p, beat_times; label="beats", lw=2)
	end
end

# ╔═╡ 5321a2d3-9c48-4e64-827c-49f83f499280
audio_player(with_clicks(beat_times), sr)

# ╔═╡ 8280d311-e4bb-4bf4-bd25-846dcf21ff5e
md"""
## 8. Harmonic / percussive separation

`Hpss(spec; kernel, power, margin, edge)` (Fitzgerald 2010) median-filters the magnitude spectrogram along **time** (sustained partials survive: harmonic) and along **frequency** (broadband clicks survive: percussive), then builds soft masks.
Both components are `DerivedSpec`s on the same grid, so they feed every downstream stage; on an STFT, `get_harmonic_signal` / `get_percussive_signal` invert them to audio.

| parameter | effect |
|:--|:--|
| `kernel = (h, p)` | median lengths in frames (harmonic) and bins (percussive). A longer time kernel demands longer sustain to count as harmonic. |
| `power` | mask exponent. `1` gives gentle masks, `2` Wiener-like, `Inf` hard binary masks (clean separation, more artefacts). |
| `margin = (h, p)` | a bin goes to a component only if it beats the other by this factor. Margins > 1 leave a residual that belongs to neither, so both components get purer. |
| `edge` | `:shrink` shrinks the median window at the borders; `:zero` pads with zeros (audioFlux, scipy). |
| `method` of the signal getters | `:wola` or `:ola` inverse STFT (see notebook 2). |
"""

# ╔═╡ f69d1766-095e-4b48-a473-bd8fc0a6f441
md"""
harmonic kernel (frames): $(@bind kh Slider(3:2:61; default=31, show_value=true))
percussive kernel (bins): $(@bind kp Slider(3:2:61; default=31, show_value=true))
power: $(@bind hpower Select([1.0 => "1", 2.0 => "2", Inf => "Inf (hard)"]; default=2.0))

margin harmonic: $(@bind mh Slider(1:0.5:5; default=1, show_value=true))
margin percussive: $(@bind mp Slider(1:0.5:5; default=1, show_value=true))
edge: $(@bind edgemode Select([:shrink, :zero]))
inverse: $(@bind invmethod Select([:wola, :ola]))
"""

# ╔═╡ 2bcc84a6-aa4f-4d4a-be2b-1e34b8fac612
hp = Hpss(stft; kernel=(kh, kp), power=hpower, margin=(mh, mp), edge=edgemode)

# ╔═╡ 110daa56-8a94-48ac-9492-bec8699e967e
plot(hp; size=(720, 500))

# ╔═╡ e89831d2-564e-4848-b0b9-68aafa946272
let (Mh, Mp) = get_masks(hp), t = get_times(stft), f = get_freq(stft)
	plot(heatmap(t, f, Mh; title="harmonic mask", c=:viridis), heatmap(t, f, Mp; title="percussive mask", c=:viridis);
	     layout=(1, 2), size=(720, 280), xguide="Time (s)", ylims=(0, 4000))
end

# ╔═╡ 2be64422-3afa-417a-88f1-b30fe5b65fb0
begin
	yh = get_harmonic_signal(hp; method=invmethod)
	yp = get_percussive_signal(hp; method=invmethod)
	md"""
	harmonic: $(audio_player(yh, sr))
	percussive: $(audio_player(yp, sr))
	"""
end

# ╔═╡ f62af430-062d-4098-b39b-6de2dbf20ed3
md"""
Because the components are ordinary spectrograms, downstream stages apply unchanged: the onsets of the percussive part and the chroma of the harmonic part.
"""

# ╔═╡ 98b907f0-f4a5-4e03-9fd9-e80b9be743bb
let perc_env = OnsetStrength(MelSpec(get_percussive(hp); nbands=64)), ch = Chroma(get_harmonic(hp))
	idx = onset_detect(perc_env; delta=deltaf)
	p1 = plot(perc_env; label="onset strength of the percussive part")
	scatter!(p1, get_times(perc_env)[idx], get_data(perc_env)[idx]; label="onsets", ms=4)
	plot(p1, plot(ch; title="chroma of the harmonic part"); layout=(2, 1), size=(720, 480))
end

# ╔═╡ e5825e41-a2ea-47d8-a3d3-df820761f0cf
md"""
## 9. Noise gates

Two gates, one in time and one in time–frequency.
A little white noise is added to the signal here so there is something to gate.

**`noisegate(x, sr; threshold, attack, release, hold)`** (MATLAB `noiseGate`) mutes samples whose level is below `threshold` dB:

| parameter | effect |
|:--|:--|
| `threshold` | level in dBFS below which the gate closes. Too high cuts quiet parts of the wanted signal. |
| `attack` | time (s) to open. Short attacks keep transients; very short ones click. |
| `release` | time (s) to close. Long releases keep decaying tails but let noise through after each sound. |
| `hold` | time (s) the gate stays open after the level drops, which avoids chattering. |

**`SpectralGate(spec; threshold, freqrange, attack, release)`** gates only the bins inside `freqrange` whose level (dB relative to the spectrogram maximum) is below `threshold`; `attack` and `release` are in frames.
It returns a `DerivedSpec`.
"""

# ╔═╡ 8b0c3f55-e5f7-46fa-8e80-8ef39e60776f
md"""
noise level (dB): $(@bind noisedb Slider(-60:5:-10; default=-35, show_value=true))

time gate threshold (dB): $(@bind ngth Slider(-60:1:0; default=-25, show_value=true))
attack (s): $(@bind ngatt Slider(0:0.005:0.2; default=0.05, show_value=true))
release (s): $(@bind ngrel Slider(0:0.01:0.5; default=0.2, show_value=true))
hold (s): $(@bind nghold Slider(0:0.01:0.3; default=0.05, show_value=true))
"""

# ╔═╡ 75828979-329a-406d-a5da-010037224664
begin
	noisy = x .+ Float32(10^(noisedb / 20)) .* randn(Float32, length(x))
	gated = noisegate(noisy, sr; threshold=ngth, attack=ngatt, release=ngrel, hold=nghold)
	let t = (0:length(x)-1) ./ sr
		plot(t, noisy; label="noisy", alpha=0.5, xguide="Time (s)")
		plot!(t, gated; label="gated", alpha=0.7)
	end
end

# ╔═╡ 0c99b706-fe23-4e51-93df-ef5313c7da47
md"""
noisy: $(audio_player(noisy, sr))
time-gated: $(audio_player(gated, sr))
"""

# ╔═╡ e78916e6-a2b0-47db-a59c-6dc3ee31d476
md"""
spectral gate threshold (dB re max): $(@bind sgth Slider(-100:5:0; default=-50, show_value=true))
band low (Hz): $(@bind sglo Slider(0:100:8000; default=0, show_value=true))
band high (Hz): $(@bind sghi Slider(0:100:8000; default=8000, show_value=true))
attack (frames): $(@bind sgatt Slider(1:10; default=1, show_value=true))
release (frames): $(@bind sgrel Slider(1:20; default=3, show_value=true))
"""

# ╔═╡ 5811ba12-728d-4889-9038-d28abd5407f8
begin
	noisy_stft = Stft(Frames(AudioFile(noisy, sr); winsize, winstep=hop))
	sgate = SpectralGate(noisy_stft; threshold=sgth, freqrange=(min(sglo, sghi), max(sglo, sghi)), attack=sgatt, release=sgrel)
	plot(plot(noisy_stft; title="noisy STFT"), plot(sgate; title="spectral gate ($(get_name(sgate)))"); layout=(2, 1), size=(720, 520))
end

# ╔═╡ 780c52b8-2cc3-483a-a41e-131e9bc68778
md"""
## 10. PCEN

`pcen(spec; gain, bias, power, time_constant, eps, b)` (Wang et al. 2017; librosa `pcen`) replaces the log of a spectrogram with an adaptive gain control: `(S / (eps + M)^gain + bias)^power - bias^power`, where `M` is `S` smoothed along time.
Stationary background (hum, hiss, a drone) is normalised away while onsets stand out.

| parameter | effect |
|:--|:--|
| `gain` | how strongly the smoothed energy `M` divides the signal (0 = no normalisation, 1 = full). |
| `bias` | offset before compression; larger values make quiet regions flatter. |
| `power` | compression exponent (0.5 ≈ square root; smaller values compress more, closer to a log). |
| `time_constant` | smoothing time of `M` in seconds. Short constants adapt fast and keep only transients. |
| `eps` | floor that avoids division by zero. |
| `b` | the smoothing coefficient itself; overrides `time_constant` when set. |
"""

# ╔═╡ 27acd49b-918d-48c1-aa1f-f688910664e1
md"""
gain: $(@bind pcgain Slider(0:0.02:1; default=0.98, show_value=true))
bias: $(@bind pcbias Slider(0:0.5:10; default=2, show_value=true))
power: $(@bind pcpower Slider(0.05:0.05:1; default=0.5, show_value=true))
`time_constant` (s): $(@bind pctc Slider(0.01:0.01:1; default=0.4, show_value=true))
eps: $(@bind pceps Select([1e-6, 1e-4, 1e-2]))
override b $(@bind pcuseb CheckBox(default=false)) b: $(@bind pcb Slider(0.001:0.001:0.5; default=0.05, show_value=true))
"""

# ╔═╡ 081d1e3c-19c8-4a92-95fd-6212163be054
let mel = MelSpec(stft; nbands=64)
	P = pcen(mel; gain=pcgain, bias=pcbias, power=pcpower, time_constant=pctc, eps=pceps, b=pcuseb ? pcb : nothing)
	plot(plot(mel; title="mel spectrogram (dB)"), plot(P; db=false, title="PCEN"); layout=(2, 1), size=(720, 520))
end

# ╔═╡ a71cd9e1-4dbf-436e-8dea-61db69eb4898
md"""
## 11. Harmonic count

`harmonic_count(stft; range, count_range)` (audioFlux `Harmonic.harmonic_count`) counts the harmonic spectral peaks of every frame of a **plain power STFT** with fixed heuristic filters (peak height, 30 Hz proximity, level within 15 dB of the loudest).
It needs fine frequency resolution: audioFlux's default is a 4096-sample Hamming window with a 1024 hop.

| parameter | effect |
|:--|:--|
| winsize | frequency resolution; with short frames close partials merge and the count drops. |
| `range` | frequency range (Hz) searched for peaks. |
| `count_range` | only peaks strictly inside this range are counted (defaults to `range`). |
"""

# ╔═╡ 99b97f20-e314-4305-b47b-3ab217a2cbb0
md"""
winsize: $(@bind hcwin Select([1024, 2048, 4096]; default=4096))
range high (Hz): $(@bind hchi Slider(500:250:8000; default=4000, show_value=true))
count range low (Hz): $(@bind hcclo Slider(27:50:2000; default=27, show_value=true))
"""

# ╔═╡ 163715a1-8fa4-4e85-9d5d-47d8e02124f1
let s = Stft(Frames(audio; winsize=hcwin, winstep=hcwin ÷ 4, type=hamming))
	hc = harmonic_count(s; range=(27, hchi), count_range=(hcclo, hchi))
	p1 = plot(s; ylims=(0, hchi), title="STFT ($(hcwin) samples)")
	p2 = plot(get_times(s), hc; seriestype=:steppost, label="harmonic count", xguide="Time (s)", ylims=(0, maximum(hc; init=1) + 1))
	plot(p1, p2; layout=(2, 1), size=(720, 480))
end

# ╔═╡ 844bf6f2-7d95-49c4-9a70-138823a62821
md"""
## Next

* **8 — Time-domain features and pitch**: RMS, energy, zero crossings, and the pitch estimators (NCF, YIN, cepstral, PEF, HPS, LHS, STFT).
* **9 — Signal processing and classic methods**: conversions, weighting, time stretch and pitch shift, NMF, HMM and Viterbi.

The full list: 01_loading_and_frames, 02_stft_and_filterbanks, 03_cepstra, 04_spectral_descriptors, 05_time_frequency, 06_discrete_wavelets_and_decompositions, 07_music_and_rhythm, 08_time_domain_and_pitch, 09_signal_processing_and_classic.
"""

# ╔═╡ Cell order:
# ╟─aa8f59a3-9171-4abf-b07b-14a9410925ea
# ╟─cabcc17f-1910-48f2-a303-5c5bf6dd4957
# ╟─4347f193-f7cf-45ee-9ab9-c90f71d24430
# ╟─d59143f4-8442-489b-b4a0-af182c90f822
# ╠═e27e48d5-9ed0-47df-9442-0530d878c91b
# ╟─b462f592-e9c7-4285-910e-7486228acb3c
# ╠═12e53ef9-b230-42f6-80fa-a47d3a3551a1
# ╟─d2bb3759-1c17-4911-b03d-cc45ce38377d
# ╟─0d08feb5-5c62-4487-9cab-c2c80a909075
# ╠═09b76ba9-c937-4655-b9ff-f26650dd42b3
# ╠═1c85d684-cf0a-4980-a556-f16667d02c16
# ╟─a3250ae7-0fe5-476d-ac6d-7ae50f0e2f76
# ╠═ef361cbe-b5eb-45a7-9132-8b764dcae0c7
# ╟─3d6770c8-f9a1-4b71-b9da-8215af26fbd4
# ╟─a3d5574a-901a-4ae8-8c4b-70df3ff518b9
# ╠═b042f159-2337-4332-9e6b-3275a3524a46
# ╠═30e1a246-0f0d-4115-bb59-8a6162be9c18
# ╟─2d6f32f0-ee01-456b-b299-bcd6119cb2c6
# ╠═9a81cb41-f97e-441c-9b0b-296888b2bf67
# ╟─6a6aa3ac-fa31-4cbe-a9b3-b774bfe0cb6c
# ╟─cc4c6159-861e-48a5-a949-3a8811d2cbc3
# ╠═71dca424-9e48-447e-a5da-f3d5b26e10d2
# ╟─40e0d280-0f71-48da-8f7d-a28511231d84
# ╟─9d9bb236-c34e-48a2-ab83-96857013fa5f
# ╠═f80eb0c3-95d0-4329-a17a-b40a8f24717a
# ╠═7cc00c72-4547-49cf-a76d-87e28008e3cf
# ╟─a17e438c-7a4d-49b5-b5d3-08f01e0a7260
# ╠═2e4f03ea-b910-4c14-9c2d-563be9de8313
# ╟─5b56ee70-3f92-4b9f-9241-1c5b4b903164
# ╠═6f6d5c67-7efe-4091-b028-be110c4c6b36
# ╟─d745f47d-4a3a-4073-bbe0-aed334e831f5
# ╟─c341bbe7-7992-4d9f-b9af-96962e92e6b8
# ╠═e0ba1427-e5a5-444d-81b6-db3b8293820b
# ╟─1f50005d-b68d-416c-b3bc-ac8757a65101
# ╟─4808b9d0-d9f7-4202-80c4-d91213aaadb3
# ╠═3e4d98f5-05a4-4352-8655-d5acc0264865
# ╠═a8e49518-9c89-48b9-9709-59dc42c9aed5
# ╟─1bbaf16a-24a0-4a11-8304-25b3e391c660
# ╟─76738ad4-3593-4f8d-8745-e50663e70b18
# ╠═e8be2ff5-c00c-4959-8863-667d29a86c4d
# ╠═5321a2d3-9c48-4e64-827c-49f83f499280
# ╟─8280d311-e4bb-4bf4-bd25-846dcf21ff5e
# ╟─f69d1766-095e-4b48-a473-bd8fc0a6f441
# ╠═2bcc84a6-aa4f-4d4a-be2b-1e34b8fac612
# ╠═110daa56-8a94-48ac-9492-bec8699e967e
# ╠═e89831d2-564e-4848-b0b9-68aafa946272
# ╠═2be64422-3afa-417a-88f1-b30fe5b65fb0
# ╟─f62af430-062d-4098-b39b-6de2dbf20ed3
# ╠═98b907f0-f4a5-4e03-9fd9-e80b9be743bb
# ╟─e5825e41-a2ea-47d8-a3d3-df820761f0cf
# ╟─8b0c3f55-e5f7-46fa-8e80-8ef39e60776f
# ╠═75828979-329a-406d-a5da-010037224664
# ╠═0c99b706-fe23-4e51-93df-ef5313c7da47
# ╟─e78916e6-a2b0-47db-a59c-6dc3ee31d476
# ╠═5811ba12-728d-4889-9038-d28abd5407f8
# ╟─780c52b8-2cc3-483a-a41e-131e9bc68778
# ╟─27acd49b-918d-48c1-aa1f-f688910664e1
# ╠═081d1e3c-19c8-4a92-95fd-6212163be054
# ╟─a71cd9e1-4dbf-436e-8dea-61db69eb4898
# ╟─99b97f20-e314-4305-b47b-3ab217a2cbb0
# ╠═163715a1-8fa4-4e85-9d5d-47d8e02124f1
# ╟─844bf6f2-7d95-49c4-9a70-138823a62821
