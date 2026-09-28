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

# ╔═╡ 836192a5-8103-4960-82a3-5e06623554b6
begin
	import Pkg
	# the notebook environment next to this file pins Audio911 (from this repository), PlutoUI and Plots
	Pkg.activate(@__DIR__)
	Pkg.instantiate()
	using Audio911, PlutoUI, Plots
	include(joinpath(@__DIR__, "common.jl"))
	gr(); default(size=(720, 300), legend=:topright, titlefontsize=10, guidefontsize=9)
end

# ╔═╡ 4b2eaf55-23c0-40ac-b78f-04af9527287f
md"""
# Audio911 · 3 — Cepstra: MFCC, GTCC, deltas and the published variants

A **cepstrum** turns a band spectrogram into a handful of decorrelated coefficients per frame.
The recipe is always the same:

```
Frames → Stft → MelSpec / BarkSpec / ErbSpec → floor → log (rectify) → DCT → keep ncoeffs → lifter → (+ log energy)
```

In Audio911 every step is a keyword of `Mfcc` (or `Gtcc` for an `ErbSpec`), so the MFCCs of MATLAB, HTK, Kaldi, librosa, ETSI and python_speech_features are all the same function with different settings.
This notebook walks through every keyword, then the deltas, the six presets, and the audioFlux-style cepstral tools (`xxcc_standard`, `Deconv`, `Cepstrogram`).
"""

# ╔═╡ a638bbc9-0776-4aa0-b6f2-1dd2eeb32aed
TableOfContents()

# ╔═╡ 60983e87-5a82-44cc-a783-c81dbf0b3339
md"""
Signal: $(@bind signal_name Select(SIGNALS))
"""

# ╔═╡ 19428d8e-085e-44ad-9171-c8acab8e82ac
audio = demo_signal(signal_name)

# ╔═╡ bd4c4ebe-dcd4-41fa-8ad0-cc9d499061f1
audio_player(audio)

# ╔═╡ 54c26093-f190-4450-95bf-1dd0a82ae5ae
md"""
## 1. The band spectrogram under the cepstrum

The cepstrum can only be as good as the band spectrogram it starts from.
Notebook 2 covers these stages in depth; here are the knobs that matter most for MFCCs.

| stage | parameter | effect on the cepstrum |
|:--|:--|:--|
| `Frames` | `winsize`, `type` | frame length and window; 20–30 ms is the speech convention |
| `Stft` | `spectrum` | `power` (`|X|²`) or `magnitude` (`|X|`); after a log this only scales the coefficients by 2 |
| `MelSpec` | `nbands` | number of bands = length of the DCT; more bands keep finer spectral detail for the upper coefficients |
| `MelSpec` | `scale` | `htk` (`2595 log10(1+f/700)`) or `slaney` (linear below 1 kHz, log above) |
| `MelSpec` | `norm` | `bandwidth` (each triangle scaled by `2/bw`), `area`, or `none_norm` — changes the tilt of the log spectrum, hence C1 |
| `MelSpec` | `domain` | `:linear` triangles in Hz or `:warped` triangles in mel |
| `MelSpec` | `freqrange` | upper limit; lower it to ignore noisy high bands |
| `MelSpec` | `win_norm` | divide by the window power, MATLAB style; shifts C0 only |

winsize: $(@bind winsize Select([256, 400, 512, 1024]; default=512))
window: $(@bind wintype Select([hamming => "hamming", hanning => "hanning", povey => "povey", rect => "rect"]))
spectrum: $(@bind spectrum Select([power => "power", magnitude => "magnitude"]))

nbands: $(@bind nbands Slider(8:2:64; default=32, show_value=true))
scale: $(@bind melscale Select([htk => "htk", slaney => "slaney"]))
norm: $(@bind melnorm Select([bandwidth => "bandwidth", area => "area", none_norm => "none"]))
domain: $(@bind domain Select([:linear, :warped]))

upper frequency (Hz): $(@bind fmax Slider(2000:500:8000; default=8000, show_value=true))
win\_norm $(@bind win_norm CheckBox(default=true))
"""

# ╔═╡ 93058aa5-7394-466a-9a76-8b4949f110e3
begin
	frames = Frames(audio; winsize, winstep=winsize ÷ 2, type=wintype)
	stft   = Stft(frames; spectrum)
	mel    = MelSpec(stft; nbands, scale=melscale, norm=melnorm, domain, win_norm,
	                 freqrange=(0, min(fmax, get_sr(audio) ÷ 2)))
end

# ╔═╡ ea12c8cf-0a0b-4ab0-afcd-78619da7225d
plot(mel; title="MelSpec, $nbands bands")

# ╔═╡ 0257fb57-17c0-4613-9ffd-616a8bac160e
md"""
## 2. `Mfcc`: every keyword

| keyword | default | effect |
|:--|:--|:--|
| `ncoeffs` | `nbands ÷ 2` | how many coefficients are kept. Low coefficients describe the broad spectral envelope (formants), high ones finer ripples. 12–13 is usual for speech. `first + ncoeffs ≤ nbands`. |
| `first` | `0` | index of the first coefficient kept. `1` drops C0 (the average log level, i.e. loudness), as HTK does. |
| `rect` | `mlog` | rectification: `mlog` (log10), `nlog` (ln), `db` (10·log10) or `cubic_root`. The logs differ only by a constant factor; `cubic_root` compresses less and is not scale-invariant. |
| `floor` / `dither` | `floatmin` | value substituted for bands below it before the log. A higher floor (e.g. `1e-8` via `dither=true`, or `1.0` on 16-bit samples in HTK) keeps silent frames from producing huge negative values. |
| `top_db` | `nothing` | clip the rectified bands to `max - top_db` (librosa's `power_to_db(top_db=80)`), meant for `rect=db`. |
| `dct` | `dct_ortho` | DCT matrix: `dct_ortho` (orthonormal), `dct_htk` (`√(2/N)` on every row, C0 larger by √2), `dct_plain` (unscaled, ETSI). Only the coefficient scale changes. |
| `lifter` | `0` | lifter length `L`: coefficient `i` is multiplied by `1 + L/2·sin(πi/L)`, boosting the mid coefficients so they have comparable ranges (22 in HTK / Kaldi). |
| `lifter_offset` | `0` | added to `i` before liftering; `1` reproduces librosa. |

ncoeffs and first are limited by the number of bands chosen above.
"""

# ╔═╡ d2971e9a-5941-421c-90c6-b27651e3cba2
md"""
first: $(@bind first_coef Select([0 => "0 (keep C0)", 1 => "1 (drop C0)"]))
ncoeffs: $(@bind ncoeffs Slider(1:nbands-1; default=min(13, nbands - 1), show_value=true))

rect: $(@bind rectf Select([mlog => "mlog (log10)", nlog => "nlog (ln)", db => "db (10 log10)", cubic_root => "cubic_root"]))
floor: $(@bind floorsel Select(["floatmin" => "floatmin (default)", "dither" => "dither (1e-8)", "1e-4" => "1e-4", "eps32" => "eps(Float32)"]))
top\_db: $(@bind top_db Select([nothing => "off", 80 => "80", 40 => "40", 20 => "20"]))

dct: $(@bind dctf Select([dct_ortho => "dct_ortho", dct_htk => "dct_htk", dct_plain => "dct_plain"]))
lifter: $(@bind lifter Slider(0:1:40; default=0, show_value=true))
lifter\_offset: $(@bind lifter_offset Select([0, 1]))

colour scale without the first row $(@bind skip_first CheckBox(default=true))
"""

# ╔═╡ a56858f5-2fe7-4662-91f5-d750626fb44d
mfcc = let
	nc = min(ncoeffs, nbands - first_coef)
	fl = floorsel == "floatmin" ? nothing : floorsel == "1e-4" ? 1e-4 : floorsel == "eps32" ? eps(Float32) : nothing
	Mfcc(mel; ncoeffs=nc, first=first_coef, rect=rectf, dither=floorsel == "dither", floor=fl,
	     top_db, dct=dctf, lifter, lifter_offset)
end

# ╔═╡ 90499e53-f3b3-4ed2-9708-a657d498d182
# heatmap of a coefficient matrix (coeffs × frames), optionally leaving the first row out of the colour range
function cepheat(M, t; first=0, skip=false, title="")
	rows = skip && size(M, 1) > 1 ? (2:size(M, 1)) : (1:size(M, 1))
	heatmap(collect(t), collect(rows .- 1 .+ first), M[rows, :]; xguide="Time (s)",
	        yguide="Coefficient", title, c=:balance, clims=maximum(abs, M[rows, :]) .* (-1, 1))
end

# ╔═╡ 119bb17d-3ab9-4841-a5e0-147ff02d56a1
cepheat(get_spec(mfcc), get_times(mfcc); first=first_coef, skip=skip_first,
        title="Mfcc: $(get_ncoeffs(mfcc)) coefficients from C$first_coef, rect=$rectf, dct=$dctf, lifter=$lifter")

# ╔═╡ 960ebd75-6b2d-4955-b123-04c94826614e
md"""
The first coefficient (C0 with `first=0`) is proportional to the mean log energy of the frame and dwarfs the others, which is why the colour scale leaves it out by default.
Switch `rect` between `mlog`, `nlog` and `db`: the pattern is identical, only the colour range changes (by factors ln 10 and 10).
`cubic_root` changes the pattern because it is not a log.

### What the coefficients encode: reconstruction from `k` coefficients

With the orthonormal DCT the coefficients can be inverted.
Keeping only the first `k` gives a **smoothed** version of the log band spectrum: few coefficients keep only the envelope (formants), more coefficients bring back the harmonic ripple.
This is exactly what `ncoeffs` throws away.

frame: $(@bind frame_idx Slider(1:length(frames); default=cld(length(frames), 2), show_value=true))
k: $(@bind krec Slider(1:nbands; default=min(13, nbands), show_value=true))
"""

# ╔═╡ 39c64464-52c7-4e57-925a-84b5d94ef982
let
	R = log10.(max.(get_spec(mel), floatmin(Float32)))         # log10 bands, as with rect=mlog
	D = dct_ortho(Float32, nbands)
	C = D * R                                                   # all nbands coefficients
	rec = D[1:krec, :]' * C[1:krec, :]                          # inverse from the first k
	f = get_freq(mel)
	plot(f, R[:, frame_idx]; label="log10 mel bands", marker=:circle, ms=2,
	     xguide="band centre (Hz)", yguide="log10", title="frame $frame_idx")
	plot!(f, rec[:, frame_idx]; label="reconstruction from $krec coefficients", lw=2)
end

# ╔═╡ f402d769-4b9a-401c-8a83-c4cb16ad1e01
md"""
### The three DCT scalings and the lifter

`dct_ortho`, `dct_htk` and `dct_plain` share the same cosine shapes; only the row scaling differs, so switching between them rescales the coefficients (C0 most of all).
The lifter curve on the right shows the gain applied to each coefficient index for the `lifter` and `lifter_offset` chosen above.
"""

# ╔═╡ 769bed7f-5000-44d5-aee9-74e96b8fb80b
let N = nbands
	p1 = plot(title="DCT row scale", xguide="row k", yguide="max |D[k, :]|")
	for d in (dct_ortho, dct_htk, dct_plain)
		D = d(Float64, N)
		plot!(p1, 0:min(N, 15)-1, vec(maximum(abs, D[1:min(N, 15), :]; dims=2)); label=string(d), marker=:circle, ms=2)
	end
	i = (0:max(ncoeffs + first_coef, 2)-1) .+ lifter_offset
	g = lifter > 0 ? 1 .+ lifter / 2 .* sin.(π .* i ./ lifter) : ones(length(i))
	p2 = plot(i .- lifter_offset, g; label="L = $lifter", marker=:circle, ms=2,
	          title="lifter gain", xguide="coefficient index")
	plot(p1, p2; layout=(1, 2))
end

# ╔═╡ 2b8b7d5e-064e-467b-9d89-c4f256e72eb7
md"""
### Log energy

An energy term can replace or accompany C0.

| keyword | choices | effect |
|:--|:--|:--|
| `energy` | `nothing`, `raw_energy`, `spectrum_energy` | `raw_energy` is `Σx²` of each frame before pre-emphasis and windowing (HTK `RAWENERGY`, Kaldi `raw_energy`); `spectrum_energy` sums the front-end spectrum bins (python_speech_features). |
| `energy_mode` | `:replace`, `:append`, `:prepend` | put the natural-log energy in place of C0 (Kaldi, psf; needs `first=0`), as an extra last coefficient (HTK `_E`, ETSI) or as an extra first one (audioFlux). |
| `energy_floor` | `eps(T)` | floor before the log; ETSI uses `exp(-50)`. It sets the value silent frames get. |

energy: $(@bind energy Select([raw_energy => "raw_energy", spectrum_energy => "spectrum_energy", nothing => "none"]))
mode: $(@bind energy_mode Select([:replace, :append, :prepend]))
energy floor: $(@bind energy_floor Select([eps(Float32) => "eps(Float32)", exp(-50) => "exp(-50)", 1e-3 => "1e-3"]))
"""

# ╔═╡ 14810bdd-ebf6-4a88-a296-40832e2de354
mfcc_e = Mfcc(mel; ncoeffs=min(13, nbands), first=0, energy, energy_mode, energy_floor)

# ╔═╡ d7f8bebf-695d-41b5-addf-d67db2cf2e8d
let M = get_spec(mfcc_e), t = collect(get_times(mfcc_e))
	row = energy === nothing ? 1 : energy_mode == :append ? size(M, 1) : 1
	c0 = get_spec(Mfcc(mel; ncoeffs=min(13, nbands), first=0))[1, :]
	p1 = plot(t, M[row, :]; label=energy === nothing ? "C0" : "log energy (row $row)", title="energy term vs C0",
	          xguide="Time (s)")
	plot!(p1, t, c0; label="C0 of the plain Mfcc", ls=:dash)
	p2 = cepheat(M, t; skip=false, title="$(size(M, 1)) rows, energy $(energy === nothing ? "off" : energy_mode)")
	plot(p1, p2; layout=(2, 1), size=(720, 480))
end

# ╔═╡ cdd197a9-81ea-43bb-9813-24e7f2622e83
md"""
## 3. `Gtcc`: gammatone cepstral coefficients

`Gtcc` is the same cepstrum computed on an `ErbSpec`, a spectrogram weighted by a **gammatone** filterbank whose bands are equally spaced on the ERB scale (MATLAB's `gtcc`).
Gammatone filters have smooth, asymmetric skirts instead of triangles, and ERB spacing is denser at low frequencies than mel.
`Gtcc` accepts every `Mfcc` keyword; `Gtcc` of anything that is not an `ErbSpec` throws.

| `ErbSpec` keyword | effect |
|:--|:--|
| `nbands` | number of gammatone channels |
| `freqrange` | lowest and highest centre frequency (design is equally spaced in ERB between them) |
| `norm` | `bandwidth`, `area` or `none_norm`, as for `MelSpec` |
| `win_norm` | window normalisation (default `true`) |

ERB bands: $(@bind erb_nbands Slider(8:2:64; default=32, show_value=true))
lowest frequency (Hz): $(@bind erb_fmin Slider(20:10:300; default=50, show_value=true))
norm: $(@bind erb_norm Select([bandwidth => "bandwidth", area => "area", none_norm => "none"]))
"""

# ╔═╡ 7dfc37d1-3e8d-432c-a68c-4135e5c1ef76
begin
	erbspec = ErbSpec(stft; nbands=erb_nbands, norm=erb_norm, freqrange=(erb_fmin, min(fmax, get_sr(audio) ÷ 2)))
	gtcc = Gtcc(erbspec; ncoeffs=min(13, erb_nbands - 1), rect=rectf, dct=dctf)
end

# ╔═╡ bbddb0c3-a02c-4d76-925a-f9acebf7de42
let
	p1 = plot(erbspec; title="ErbSpec, $erb_nbands gammatone bands")
	p2 = cepheat(get_spec(gtcc), get_times(gtcc); skip=true, title="Gtcc (C0 out of the colour range)")
	p3 = cepheat(get_spec(Mfcc(mel; ncoeffs=min(13, nbands - 1), rect=rectf, dct=dctf)), get_times(mel); skip=true, title="Mfcc on the mel bands above")
	plot(p1, p2, p3; layout=(3, 1), size=(720, 700))
end

# ╔═╡ d0501110-9791-41c1-b7f6-9a4d776ba543
md"""
## 4. `Delta` and delta-delta

`Delta(x; delta_length, source)` is MATLAB's `audioDelta` regression filter, applied to any cepstrum, delta or spectrogram.
It is causal with zero initial state, so the first frames ramp up.
Applying it twice gives delta-delta (acceleration); `Audio911.DeltaDelta(x)` returns both at once.

| keyword | default | effect |
|:--|:--|:--|
| `delta_length` | `9` | odd regression window (> 2). Longer windows give smoother, more delayed derivatives; 5 corresponds to HTK / Kaldi `DELTAWINDOW 2`. |
| `source` | `:standard` | `:standard` differentiates along time (MATLAB); `:transposed` along the coefficient axis (audioFlux's convention). |

delta\_length: $(@bind delta_length Select([3, 5, 7, 9, 11, 15, 21]; default=9))
source: $(@bind delta_source Select([:standard, :transposed]))
coefficient to trace: $(@bind trace_coef Slider(1:min(13, nbands - 1); default=2, show_value=true))
"""

# ╔═╡ e8a62c9e-5d63-43d4-8e18-7ee15bbc8819
begin
	base_mfcc = Mfcc(mel; ncoeffs=min(13, nbands - 1))
	d1, d2 = Audio911.DeltaDelta(base_mfcc; delta_length, source=delta_source)
end

# ╔═╡ f3461f62-8b73-48e6-9401-62ca8df949b4
let t = collect(get_times(base_mfcc)), r = trace_coef + 1
	p1 = plot(t, get_spec(base_mfcc)[r, :]; label="C$trace_coef", title="one coefficient and its derivatives")
	plot!(p1, t, get_spec(d1)[r, :]; label="Δ")
	plot!(p1, t, get_spec(d2)[r, :]; label="ΔΔ")
	p2 = cepheat(get_spec(d1), t; skip=true, title="Δ (delta_length=$delta_length, $delta_source)")
	p3 = cepheat(get_spec(d2), t; skip=true, title="ΔΔ")
	plot(p1, p2, p3; layout=(3, 1), size=(720, 700), xguide="Time (s)")
end

# ╔═╡ f984e7e9-ee94-4f4d-a8ca-1793b754d878
md"""
## 5. The published MFCC variants side by side

Six presets wire the keywords above to reproduce reference implementations.
Each is a plain composition of `Frames`, `Stft`, `MelSpec` and `Mfcc` that you can copy and change.

| preset | framing | window | spectrum | bands | log | DCT | coeffs | lifter | energy |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `mfcc_matlab` | 30 ms / 10 ms hop | periodic Hamming | power, window-normalised | 32 HTK-mel, bandwidth norm | log10 | ortho | 13 from C0 | – | – |
| `mfcc_htk` | 25 / 10 ms, pre-emph 0.97 | symmetric Hamming | magnitude, ×32768 | 20 mel-domain triangles | ln, floor 1 | `dct_htk` | 12 from C1 | 22 | `energy=true` appends `_E`, `c0=true` keeps C0 |
| `mfcc_kaldi` | 25 / 10 ms, DC removal, pre-emph 0.97 | Povey | power | 23 mel from 20 Hz | ln, floor eps | ortho | 13 | 22 | raw log energy replaces C0 |
| `mfcc_librosa` | centred, 2048 / 512 | periodic Hann | power | 128 Slaney-mel | 10 log10, top 80 dB | ortho | 20 | – (`lifter_offset=1`) | – |
| `mfcc_etsi` | offset compensation, 25 / 10 ms, pre-emph 0.97 | symmetric Hamming | magnitude, ×32768 | 23 integer-bin (`etsi_fbank`) from 64 Hz | ln, floor 2e-22 | plain | 13 from C0 | – | raw log energy appended |
| `mfcc_psf` | signal pre-emph 0.97, 25 / 10 ms, last frame padded | rect | power / nfft | 26 integer-bin (`psf_fbank`) | ln | ortho | 13 | 22 | spectrum log energy replaces C0 |

Shared keywords every preset accepts: `winsize`, `winstep`, `nfft`, `nbands`, `ncoeffs`, `freqrange`; most also take `preemph`, `lifter` and `energy`, and HTK / ETSI take `scale` (the 16-bit sample scale).
Override them below (a value of 0 keeps each preset's own default).

nbands override: $(@bind v_nbands Slider(0:2:64; default=0, show_value=true))
ncoeffs override: $(@bind v_ncoeffs Slider(0:1:20; default=0, show_value=true))
HTK energy $(@bind v_htk_energy CheckBox(default=false)) HTK c0 $(@bind v_htk_c0 CheckBox(default=false))
Kaldi energy $(@bind v_kaldi_energy CheckBox(default=true)) psf energy $(@bind v_psf_energy CheckBox(default=true))
librosa top\_db: $(@bind v_top_db Select([80 => "80", 40 => "40", nothing => "off"]))
"""

# ╔═╡ ede55792-2e7f-4230-9f70-b68129c88269
variants = let
	# each preset's own (nbands, ncoeffs) defaults; overrides are clamped so first + ncoeffs ≤ nbands
	presets = [
		("matlab",  mfcc_matlab,  32,  13, 0, (;)),
		("htk",     mfcc_htk,     20,  12, 1, (; energy=v_htk_energy, c0=v_htk_c0)),
		("kaldi",   mfcc_kaldi,   23,  13, 0, (; energy=v_kaldi_energy)),
		("librosa", mfcc_librosa, 128, 20, 0, (; top_db=v_top_db)),
		("etsi",    mfcc_etsi,    23,  13, 0, (;)),
		("psf",     mfcc_psf,     26,  13, 0, (; energy=v_psf_energy)),
	]
	map(presets) do (name, f, nb0, nc0, extra, kw)
		nb = v_nbands > 0 ? v_nbands : nb0
		nc = min(v_ncoeffs > 0 ? v_ncoeffs : nc0, nb - extra)   # HTK keeps c1.. (or c0 plus one more)
		name => f(audio; nbands=nb, ncoeffs=nc, kw...)
	end
end

# ╔═╡ 5faeeb3c-7887-40a3-a173-0cebfb70b800
plot([cepheat(get_spec(m), get_times(m); skip=true, title=name) for (name, m) in variants]...;
     layout=(3, 2), size=(760, 760), titlefontsize=9, guidefontsize=7, tickfontsize=6, colorbar=false)

# ╔═╡ 7b9afc49-ef99-44c9-a072-fe2649fb2df4
[(preset=name, frames=size(get_data(m), 1), coeffs=size(get_data(m), 2),
  C1_range=round.(extrema(get_spec(m)[min(2, end), :]); sigdigits=3)) for (name, m) in variants]

# ╔═╡ 986f6fc0-2ca5-471a-ac86-ddeb887ce34f
md"""
The patterns agree while the numbers do not: frame counts differ (librosa's 512-sample hop and centring, psf's padded last frame), coefficient scales differ (HTK's and ETSI's 16-bit scaling and DCT scaling), and energy rows appear or replace C0.
When comparing features with another toolkit, pick its preset rather than tuning a generic `Mfcc` by hand.

### Integer-bin filterbanks: `etsi_fbank` and `psf_fbank`

ETSI and python_speech_features build their triangles on rounded FFT bins instead of continuous frequencies.
`etsi_fbank(sr; nfft, nbands=23, freqrange=(64, sr÷2))` and `psf_fbank(sr; nfft, nbands=26, freqrange=(0, sr÷2))` return ordinary `FBank`s you can pass to `MelSpec(stft, fbank)`.
The rounding is visible at low frequencies with a small `nfft`, where adjacent triangles collapse onto one or two bins.

nfft: $(@bind fb_nfft Select([256, 512, 1024]; default=256))
bands: $(@bind fb_nbands Slider(10:1:40; default=23, show_value=true))
"""

# ╔═╡ 1a988200-5501-4762-8f3c-c8a185fc911c
let sr = get_sr(audio)
	fe = etsi_fbank(sr; nfft=fb_nfft, nbands=fb_nbands)
	fp = psf_fbank(sr; nfft=fb_nfft, nbands=fb_nbands)
	fa = auditory_fbank(sr; nfft=fb_nfft, nbands=fb_nbands, scale=htk, norm=none_norm, freqrange=(64, sr ÷ 2))
	plot(plot(fe; title="etsi_fbank", legend=false), plot(fp; title="psf_fbank", legend=false),
	     plot(fa; title="auditory_fbank (continuous triangles)", legend=false);
	     layout=(3, 1), size=(720, 640), xlims=(0, 2500))
end

# ╔═╡ cecbe267-283e-4181-a14c-68c7eb2e4e45
md"""
### `offset_compensation`

ETSI's DC notch filter `y[n] = x[n] - x[n-1] + coef·y[n-1]` removes a constant offset from the whole signal before framing.
`coef` close to 1 gives a very narrow notch at 0 Hz; smaller values also attenuate the lowest audible frequencies.
Below, a DC offset is added to the signal and removed again.

coef: $(@bind oc_coef Slider(0.9:0.001:0.999; default=0.999, show_value=true))
added offset: $(@bind dc_offset Slider(0:0.05:0.5; default=0.3, show_value=true))
"""

# ╔═╡ 600f4305-030f-4f9b-b78c-fdd34ea0d4f8
let x = vec(get_data(audio)) .+ Float32(dc_offset), sr = get_sr(audio)
	y = offset_compensation(x; coef=oc_coef)
	t = (0:length(x)-1) ./ sr
	plot(t, x; label="signal + offset (mean $(round(sum(x) / length(x); digits=3)))", alpha=0.7, xguide="Time (s)")
	plot!(t, y; label="offset_compensation (mean $(round(sum(y) / length(y); digits=3)))", alpha=0.7)
end

# ╔═╡ 58817660-52aa-4267-a0e8-ae564629b207
md"""
## 6. audioFlux-style cepstral tools

### `xxcc_standard`

`xxcc_standard(spec; ncoeffs=13, energy=spectrum_energy, energy_mode=:replace, delta_length=9, rect=mlog)` returns `(cepstrum, delta, deltadelta)` exactly as audioFlux's `xxcc_standard` / `mfcc_standard`: an `Mfcc` with a `1e-8` floor and the orthonormal DCT, a log energy, and deltas taken along the **coefficient** axis (`source=:transposed`).
It accepts any spectrogram, so the same call gives audioFlux's `cqcc`, `mfcc`, `bfcc` or `gtcc` depending on its input.

| keyword | effect |
|:--|:--|
| `ncoeffs` | coefficients kept |
| `energy` | `spectrum_energy`, `raw_energy`, a vector of one value per frame, or `nothing` |
| `energy_mode` | `:replace` C0 or `:prepend` the energy (audioFlux's `APPEND`) |
| `delta_length` | regression window of both deltas |
| `rect` | rectification, as for `Mfcc` |

input: $(@bind xx_input Select(["stft" => "Stft (magnitude)", "mel" => "MelSpec above", "erb" => "ErbSpec above"]))
ncoeffs: $(@bind xx_ncoeffs Slider(2:1:20; default=13, show_value=true))
energy: $(@bind xx_energy Select([spectrum_energy => "spectrum_energy", raw_energy => "raw_energy", nothing => "none"]))
mode: $(@bind xx_mode Select([:replace, :prepend]))
delta\_length: $(@bind xx_dl Select([3, 5, 9, 15]; default=9))
"""

# ╔═╡ c6e84cdc-d079-4fb7-87ea-812ccf7d47e3
xx = let s = xx_input == "stft" ? Stft(frames; spectrum=magnitude) : xx_input == "mel" ? mel : erbspec
	nc = min(xx_ncoeffs, get_nbins(s) - 1)
	xxcc_standard(s; ncoeffs=nc, energy=xx_energy, energy_mode=xx_mode, delta_length=xx_dl)
end

# ╔═╡ f291c0c0-cfdf-4027-b95e-d3c6b21c613c
plot([cepheat(get_spec(m), get_times(m); skip=true, title=n) for (n, m) in zip(("cepstrum", "Δ (along coefficients)", "ΔΔ"), xx)]...;
     layout=(3, 1), size=(720, 700))

# ╔═╡ bfa88558-2aaf-4ac5-9e3e-2466bdf79f34
md"""
### `Deconv`: timbre and pitch

`Deconv(spec)` (audioFlux `deconv`) splits every frame of a spectrogram into a smooth **timbre** envelope (`get_timbre`) and the fine **pitch** structure (`get_pitch`, the harmonic comb), by Fourier-transforming the frame's bins and separating magnitude from phase.
It has no keywords; audioFlux applies it to a magnitude spectrogram, and the choice of input spectrogram is the knob.

input spectrum: $(@bind dc_spectrum Select([magnitude => "magnitude", power => "power"]))
frame: $(@bind dc_frame Slider(1:length(frames); default=cld(length(frames), 2), show_value=true))
"""

# ╔═╡ a52c560f-0c70-4d76-86a6-2b7c66ff70dd
deconv = Deconv(Stft(frames; spectrum=dc_spectrum))

# ╔═╡ f0b465bb-999e-4b87-abc0-7b0430103d84
let f = collect(get_freq(deconv))
	p1 = plot(deconv)
	p2 = plot(f, get_spec(get_parent(deconv))[:, dc_frame] ./ maximum(get_spec(get_parent(deconv))[:, dc_frame]);
	          label="spectrum", title="frame $dc_frame (each curve scaled to its maximum)", xguide="Hz", alpha=0.6)
	tm = get_timbre(deconv)[:, dc_frame]; pt = get_pitch(deconv)[:, dc_frame]
	plot!(p2, f, tm ./ maximum(abs, tm); label="timbre", lw=2)
	plot!(p2, f, pt ./ maximum(abs, pt); label="pitch", alpha=0.6)
	plot(p1, p2; layout=@layout([a{0.62h}; b]), size=(720, 720))
end

# ╔═╡ 6343c4a5-972c-4e52-b052-6894fc13d4de
md"""
### `Cepstrogram`: cepstrum, envelope and details

`Cepstrogram(frames; ncep=4)` (audioFlux) computes the real cepstrum of each frame, `real(ifft(log P))`, against **quefrency** (`get_quefrency`, in seconds).
Keeping quefrencies `0:ncep` and transforming back gives the smooth spectral **envelope** (`get_envelope`); the rest gives the **details** (`get_details`).
A voiced frame shows a cepstral peak at the pitch period — for speech, somewhere between 2.5 and 12 ms.

| keyword | effect |
|:--|:--|
| `ncep` | lifter cut-off (1 to `winsize ÷ 2 - 1`). Small values give a very smooth envelope; large values let pitch harmonics leak into it. |
| frame `type` | audioFlux uses a rectangular window (`type=rect`); a tapered window gives a cleaner cepstrum. |

ncep: $(@bind ncep Slider(1:1:80; default=20, show_value=true))
window: $(@bind cg_win Select([rect => "rect (audioFlux)", hanning => "hanning", hamming => "hamming"]))
frame: $(@bind cg_frame Slider(1:length(frames); default=cld(length(frames), 2), show_value=true))
"""

# ╔═╡ 1a0cefb5-6b58-4dcf-b6d8-d3c409f378ea
cepgram = Cepstrogram(Frames(audio; winsize, winstep=winsize ÷ 2, type=cg_win); ncep=min(ncep, winsize ÷ 2 - 1))

# ╔═╡ 3a0a882d-8c3e-45aa-a3f6-2edd86bde658
let q = get_quefrency(cepgram), f = get_freq(cepgram), j = min(cg_frame, get_nframes(cepgram))
	C = get_data(cepgram)
	keep = 2:min(length(q), round(Int, 0.02 * get_sr(audio)))    # skip quefrency 0, show up to 20 ms
	p1 = heatmap(collect(get_times(cepgram)), 1000 .* q[keep], C[keep, :]; c=:balance,
	             clims=maximum(abs, C[keep, :]) .* (-1, 1), title="cepstrum", yguide="quefrency (ms)", xguide="Time (s)")
	p2 = plot(1000 .* q[keep], C[keep, j]; label="frame $j", xguide="quefrency (ms)", title="cepstrum of one frame")
	vline!(p2, [1000 * q[ncep + 1]]; label="ncep", ls=:dash)
	p3 = plot(f, get_envelope(cepgram)[:, j] .+ get_details(cepgram)[:, j]; label="log power", alpha=0.5,
	          xguide="Hz", title="envelope and details, frame $j")
	plot!(p3, f, get_envelope(cepgram)[:, j]; label="envelope (ncep=$ncep)", lw=2)
	plot(p1, p2, p3; layout=(3, 1), size=(720, 760))
end

# ╔═╡ f141c51a-4907-4186-87c5-3460d5b8128d
md"""
## Next

* **4 — Spectral descriptors**: one value per frame (centroid, flatness, flux, rolloff, …) on any spectrogram.
* **5 — Time-frequency transforms**: wavelets, CQT, Stockwell, NSGT, synchrosqueezing, reassignment, Wigner–Ville — all of which can feed the same `MelSpec` → `Mfcc` chain shown here.

The full tour: 01 loading and frames · 02 STFT and filterbanks · 03 cepstra · 04 spectral descriptors · 05 time-frequency · 06 discrete wavelets and decompositions · 07 music and rhythm · 08 time domain and pitch · 09 signal processing and classic.
"""

# ╔═╡ Cell order:
# ╟─4b2eaf55-23c0-40ac-b78f-04af9527287f
# ╟─836192a5-8103-4960-82a3-5e06623554b6
# ╟─a638bbc9-0776-4aa0-b6f2-1dd2eeb32aed
# ╟─60983e87-5a82-44cc-a783-c81dbf0b3339
# ╠═19428d8e-085e-44ad-9171-c8acab8e82ac
# ╠═bd4c4ebe-dcd4-41fa-8ad0-cc9d499061f1
# ╟─54c26093-f190-4450-95bf-1dd0a82ae5ae
# ╠═93058aa5-7394-466a-9a76-8b4949f110e3
# ╠═ea12c8cf-0a0b-4ab0-afcd-78619da7225d
# ╟─0257fb57-17c0-4613-9ffd-616a8bac160e
# ╟─d2971e9a-5941-421c-90c6-b27651e3cba2
# ╠═a56858f5-2fe7-4662-91f5-d750626fb44d
# ╠═90499e53-f3b3-4ed2-9708-a657d498d182
# ╠═119bb17d-3ab9-4841-a5e0-147ff02d56a1
# ╟─960ebd75-6b2d-4955-b123-04c94826614e
# ╠═39c64464-52c7-4e57-925a-84b5d94ef982
# ╟─f402d769-4b9a-401c-8a83-c4cb16ad1e01
# ╠═769bed7f-5000-44d5-aee9-74e96b8fb80b
# ╟─2b8b7d5e-064e-467b-9d89-c4f256e72eb7
# ╠═14810bdd-ebf6-4a88-a296-40832e2de354
# ╠═d7f8bebf-695d-41b5-addf-d67db2cf2e8d
# ╟─cdd197a9-81ea-43bb-9813-24e7f2622e83
# ╠═7dfc37d1-3e8d-432c-a68c-4135e5c1ef76
# ╠═bbddb0c3-a02c-4d76-925a-f9acebf7de42
# ╟─d0501110-9791-41c1-b7f6-9a4d776ba543
# ╠═e8a62c9e-5d63-43d4-8e18-7ee15bbc8819
# ╠═f3461f62-8b73-48e6-9401-62ca8df949b4
# ╟─f984e7e9-ee94-4f4d-a8ca-1793b754d878
# ╠═ede55792-2e7f-4230-9f70-b68129c88269
# ╠═5faeeb3c-7887-40a3-a173-0cebfb70b800
# ╠═7b9afc49-ef99-44c9-a072-fe2649fb2df4
# ╟─986f6fc0-2ca5-471a-ac86-ddeb887ce34f
# ╠═1a988200-5501-4762-8f3c-c8a185fc911c
# ╟─cecbe267-283e-4181-a14c-68c7eb2e4e45
# ╠═600f4305-030f-4f9b-b78c-fdd34ea0d4f8
# ╟─58817660-52aa-4267-a0e8-ae564629b207
# ╠═c6e84cdc-d079-4fb7-87ea-812ccf7d47e3
# ╠═f291c0c0-cfdf-4027-b95e-d3c6b21c613c
# ╟─bfa88558-2aaf-4ac5-9e3e-2466bdf79f34
# ╠═a52c560f-0c70-4d76-86a6-2b7c66ff70dd
# ╠═f0b465bb-999e-4b87-abc0-7b0430103d84
# ╟─6343c4a5-972c-4e52-b052-6894fc13d4de
# ╠═1a0cefb5-6b58-4dcf-b6d8-d3c409f378ea
# ╠═3a0a882d-8c3e-45aa-a3f6-2edd86bde658
# ╟─f141c51a-4907-4186-87c5-3460d5b8128d
