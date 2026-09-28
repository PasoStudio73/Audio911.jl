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

# ╔═╡ a7af9b36-a52f-43d4-b5e9-097373ee2cd1
begin
	import Pkg
	# the notebook environment next to this file pins Audio911 (from this repository), PlutoUI and Plots
	Pkg.activate(@__DIR__)
	Pkg.instantiate()
	using Audio911, PlutoUI, Plots
	include(joinpath(@__DIR__, "common.jl"))
	gr(); default(size=(720, 300), legend=:topright, titlefontsize=10, guidefontsize=9)
end

# ╔═╡ e49abb6e-7797-47c9-b6e1-4252128cc75b
md"""
# Audio911 · 2 — STFT and filterbanks

```
AudioFile → Frames → Stft → FBank → LinSpec / MelSpec / BarkSpec / ErbSpec
```

This notebook turns frames into a **short-time Fourier transform**, inverts it back to sound with **`istft`**, designs **filterbanks** on every supported scale, and applies them to obtain linear, mel, bark and ERB spectrograms.
The last section shows that the same filterbank stages run unchanged on a wavelet front end (`Cwt`).

Framing parameters (`winsize`, `winstep`, window type, …) are covered in notebook 1; only the most important ones are repeated here.
"""

# ╔═╡ e5f1a706-fd08-442c-93a7-d624a331b438
TableOfContents()

# ╔═╡ d8107959-f134-4ac4-841d-5df066bf0472
md"""
## 0. Signal and frames

Signal: $(@bind signal_name Select(SIGNALS))
winsize: $(@bind winsize Select([128, 256, 400, 512, 1024, 2048]; default=512))
hop: $(@bind hopfrac Select([0.5 => "1/2", 0.25 => "1/4", 0.125 => "1/8"]; default=0.25))
window: $(@bind wintype Select([hanning => "hanning", hamming => "hamming", blackman => "blackman", rect => "rect", povey => "povey"]))
center $(@bind center CheckBox(default=false))
"""

# ╔═╡ d4cd84c8-c891-4880-8585-19796b4712f4
begin
	audio = demo_signal(signal_name)
	sr = get_sr(audio)
	frames = Frames(audio; winsize, winstep=round(Int, winsize * hopfrac), type=wintype, center)
end

# ╔═╡ 5cb2027b-9430-422e-9561-7b17a5f13bc9
audio_player(audio)

# ╔═╡ 627cd53e-692e-46a5-8bac-df8b6669577d
md"""
## 1. Short-time Fourier transform — `Stft`

`Stft(frames; nfft, spectrum, scale, keep_complex)` windows every frame, zero-pads it to `nfft`, runs a real FFT and keeps the one-sided `nfft ÷ 2 + 1` bins.
The result is stored **bins × frames**.
`Stft(audio; winsize, winstep, type, …, nfft, spectrum, …)` does framing and transform in one call and accepts every `Frames` keyword.

| parameter | effect |
|:--|:--|
| `nfft` | FFT size, `≥ winsize`. Bin spacing becomes `sr / nfft`. Zero padding **interpolates** the spectrum (smoother curve, more bins) but does not separate tones closer than `sr / winsize`: only a longer window does that. |
| `spectrum` | `power` (`|X|²`, default) or `magnitude` (`|X|`). Power emphasises the loud components; magnitude is closer to perceived loudness in dB terms (20 log vs 10 log give the same dB picture). |
| `scale` | constant multiplier of the spectrogram. `1/nfft` reproduces python_speech_features' `powspec`. Only the absolute level changes. |
| `keep_complex` | also store the complex STFT, so `get_complex` / `get_phase` and `istft` do not recompute it. The values are identical either way; this is a memory-for-time trade. |

The plot recipe takes `db` (dB relative to the maximum), `top_db` (dynamic range shown) and `freq_scale` (`:linear` or `:log10`).
"""

# ╔═╡ f7d606db-c606-44e8-99d4-d9eefb6289a5
md"""
nfft: $(@bind nfft_mult Select([1 => "= winsize", 2 => "2 × winsize", 4 => "4 × winsize"]))
spectrum: $(@bind spectrum_kind Select([power => "power", magnitude => "magnitude"]))
scale: $(@bind stft_scale Select([1.0 => "1", -1.0 => "1/nfft"]))
keep complex $(@bind keep_complex CheckBox(default=true))

plot in dB $(@bind show_db CheckBox(default=true))
top\_db: $(@bind top_db Slider(20:10:120; default=80, show_value=true))
frequency axis: $(@bind freq_scale Select([:linear, :log10]))
"""

# ╔═╡ c32f544a-7678-4532-b24f-fcaaeb9e0a8a
stft = let nfft = nfft_mult * winsize
	Stft(frames; nfft, spectrum=spectrum_kind, scale=stft_scale < 0 ? 1 / nfft : stft_scale, keep_complex)
end

# ╔═╡ 825cfb11-636d-4ce9-baae-049fb0a31022
plot(stft; db=show_db, top_db, freq_scale)

# ╔═╡ cb046166-8b2e-4aad-9fb7-cb23f3dfe1bf
(bins = get_nbins(stft), frames = get_nframes(stft), nfft = get_nfft(stft),
 bin_spacing_Hz = sr / get_nfft(stft), resolution_Hz = sr / winsize,
 max_value = maximum(get_spec(stft)))

# ╔═╡ e349e3e9-28d9-4c2f-bb4e-df8cfc1d7c48
md"""
### One frame, zoomed in

The slice below shows a single column.
With the *two close tones* signal (440 + 470 Hz), a 512-sample window at 16 kHz (31 Hz resolution) barely separates them: increasing `nfft` only makes the curve smoother, while doubling `winsize` in section 0 actually splits the two peaks.
"""

# ╔═╡ 143061a1-2349-48f9-809a-c8c48d9010db
@bind stft_frame Slider(1:get_nframes(stft); default=cld(get_nframes(stft), 2), show_value=true)

# ╔═╡ 9b36bfbe-1a58-4a1d-8783-5611752a4aff
@bind fmax_zoom Slider(200:100:sr÷2; default=1500, show_value=true)

# ╔═╡ fef5910d-1f4d-43a0-becc-47fb46ed327c
let S = get_spec(stft)[:, stft_frame], f = get_freq(stft)
	k = f .<= fmax_zoom
	dB = (spectrum_kind === power ? 10 : 20) .* log10.(max.(S ./ maximum(get_spec(stft)), 1e-12))
	plot(f[k], dB[k]; marker=(:circle, 2), xguide="Hz", yguide="dB", label="frame $stft_frame",
	     title="nfft = $(get_nfft(stft)), bin spacing $(round(sr / get_nfft(stft); digits=1)) Hz", ylims=(-100, 5))
end

# ╔═╡ 7349946f-7781-422d-852d-23c145b0e5c2
md"""
### Phase

`get_complex(stft)` returns the complex one-sided STFT and `get_phase(stft)` its angle.
Raw phase looks like noise; its **frame-to-frame difference** is what carries instantaneous-frequency information (used by the phase vocoder in notebook 9 and by reassignment in notebook 5).
"""

# ╔═╡ b8de0ac7-8fed-48a0-82ae-61411b891754
let ϕ = get_phase(stft), f = get_freq(stft), t = get_times(stft)
	dϕ = mod2pi.(diff(ϕ; dims=2) .+ π) .- π
	p1 = heatmap(t, f, ϕ; title="phase", c=:twilight, xguide="Time (s)", yguide="Hz")
	p2 = heatmap(t[2:end], f, dϕ; title="phase difference between frames", c=:twilight, xguide="Time (s)")
	plot(p1, p2; layout=(1, 2), size=(720, 300))
end

# ╔═╡ 162e1fde-b3c0-4cab-bcca-6b96eb0edeef
md"""
## 2. Inverse STFT — `istft`

`istft(stft; method, length)` rebuilds the signal the frames were cut from.
`istft(C, winsize, winstep; window, periodic, nfft, method, offset, length)` inverts any complex matrix, so you can **edit the STFT and listen to the result**.

| parameter | effect |
|:--|:--|
| `method` | `:wola` (weighted overlap-add, the default of librosa, MATLAB and audioFlux): every frame is windowed again and the sum divided by `Σ w²`, robust when the STFT was modified. `:ola`: plain overlap-add divided by `Σ w`, exact only for an unmodified STFT. |
| `length` | truncate or zero-pad the output. |
| `window`, `periodic`, `nfft` | (matrix form) must match the analysis to reconstruct exactly. |
| `offset` | (matrix form) index of the first sample of frame 1 relative to the signal; negative for centred frames. |

Inside the signal the reconstruction is exact to Float32 precision for every window and hop offered here.
At the two ends only the tapered edge of a single window covers the samples, so they come back attenuated or wrong unless the frames are centred: tick **center** in section 0 and the whole-signal error drops to the same level.
Below, a brick-wall mask keeps only bins between two cut-offs before inverting.
"""

# ╔═╡ f533ad7e-52cb-4ab1-8bf5-cb27efea6eaf
md"""
method: $(@bind istft_method Select([:wola, :ola]))
keep from $(@bind mask_lo Slider(0:50:sr÷2; default=0, show_value=true)) Hz
to $(@bind mask_hi Slider(50:50:sr÷2; default=sr÷2, show_value=true)) Hz
"""

# ╔═╡ e5ee623a-54ec-4f95-8639-7b1d80a7c1d0
begin
	y_rec = istft(stft; method=istft_method, length=length(get_data(audio)))
	x_ref = vec(get_data(audio))
	# away from the edges every sample is covered by enough overlapping windows
	inner = winsize+1:length(x_ref)-winsize
	(max_error_inside = maximum(abs, y_rec[inner] .- x_ref[inner]),
	 max_error_whole_signal = maximum(abs, y_rec .- x_ref),
	 rms_signal = sqrt(sum(abs2, x_ref) / length(x_ref)))
end

# ╔═╡ 4904640f-1df0-45a6-941d-2e7fd7eaa4c4
y_masked = let C = copy(get_complex(stft)), f = get_freq(stft)
	C[(f .< mask_lo) .| (f .> mask_hi), :] .= 0
	istft(C, winsize, get_step(stft); window=wintype, nfft=get_nfft(stft), method=istft_method,
	      offset=get_offset(stft), length=length(x_ref))
end

# ╔═╡ 33d0eb1e-8777-4d58-8f43-7cd2b8543348
let t = (0:length(x_ref)-1) ./ sr
	plot(t, x_ref; label="original", alpha=0.6, xguide="Time (s)")
	plot!(t, y_masked; label="masked $(mask_lo)–$(mask_hi) Hz", alpha=0.7)
end

# ╔═╡ 5194f139-0c89-45ac-bafc-b7d67916cb96
md"Original → reconstructed → masked: $(audio_player(x_ref, sr)) $(audio_player(y_rec, sr)) $(audio_player(y_masked, sr))"

# ╔═╡ a337406d-eb97-47fa-bd84-96bba3e90deb
md"""
## 3. Filterbank design — `auditory_fbank` and `gammatone_fbank`

A filterbank is a `nbands × nbins` weight matrix evaluated on a frequency grid: here the STFT grid (`auditory_fbank(stft; …)` reads `sr` and the grid from the spectrogram).

| parameter | effect |
|:--|:--|
| `nbands` | number of filters. More bands → narrower filters and finer spectral detail; too many for a small `nfft` leaves some filters covering one bin or none. |
| `scale` | where the centres go. `htk` and `slaney` are mel scales (dense below 1 kHz, sparse above); `bark` follows critical bands; `erb` the Glasberg–Moore ERB-rate; `linspace` is uniform; `logspace` geometric; `octave` a musical grid of `bins_per_octave` per octave anchored on 440 Hz (the upper end of `freqrange` is ignored). |
| `norm` | `bandwidth` divides by half the bandwidth (every band has the same area-per-Hz, librosa's Slaney norm); `area` divides by the sum of weights (every band sums to 1); `none_norm` keeps peak 1, so wide high bands collect more energy. |
| `domain` | `:linear` draws straight triangles in Hz; `:warped` draws them straight on the warped scale, so they look curved in Hz. |
| `style` | band shape: `triangular` (default), `etsi` (audioFlux ETSI triangle on nearest bins), `point` (a single bin), or a window (`rect`, `hanning`, `hamming`, `blackman`, `bohman`, `kaiser`, `gauss`). Window styles are designed in the linear domain only. |
| `freqrange` | `(low, high)` Hz covered by the bank. Cutting the low end removes hum; cutting the top removes hiss. |
| `bins_per_octave` | only for `scale=octave`. |
| `nfft`, `sfreq` | the grid, when designing without a spectrogram. |

`gammatone_fbank(stft; nbands, norm, freqrange)` designs ERB-spaced gammatone filters instead (used by `ErbSpec`).
"""

# ╔═╡ 2e48dd93-3d60-426c-9c06-3bcc8e166701
md"""
scale: $(@bind fb_scale Select([htk => "htk (mel)", slaney => "slaney (mel)", bark => "bark", erb => "erb", linspace => "linspace", logspace => "logspace", octave => "octave"]))
nbands: $(@bind fb_nbands Slider(4:2:80; default=26, show_value=true))
norm: $(@bind fb_norm Select([bandwidth => "bandwidth", area => "area", none_norm => "none_norm"]))

domain: $(@bind fb_domain Select([:linear, :warped]))
style: $(@bind fb_style Select([triangular => "triangular", etsi => "etsi", point => "point", rect => "rect", hanning => "hanning", hamming => "hamming", blackman => "blackman", bohman => "bohman", kaiser => "kaiser", gauss => "gauss"]))
bins per octave (octave scale): $(@bind fb_bpo Slider(4:48; default=12, show_value=true))

low: $(@bind fb_lo Slider(0:10:2000; default=50, show_value=true)) Hz
high: $(@bind fb_hi Slider(1000:250:sr÷2; default=7000, show_value=true)) Hz
"""

# ╔═╡ ec0fa127-1368-4ca3-91b4-135b664d273f
"""
    try_design(f, fallback) -> (result, message)

Run a filterbank design; when the chosen parameters are impossible (for example an
octave grid running past Nyquist, or a window style in the warped domain), return
`fallback()` and the library's error message instead so the notebook keeps working.
"""
function try_design(f, fallback)
	try
		return f(), nothing
	catch e
		e isa ArgumentError || rethrow()
		return fallback(), e.msg
	end
end

# ╔═╡ 55e01f2d-eb02-4647-aca8-eaec2420890e
fbank, fb_msg = try_design(
	() -> auditory_fbank(stft; nbands=fb_nbands, scale=fb_scale, norm=fb_norm, domain=fb_domain,
	                     style=fb_style, freqrange=(fb_lo, fb_hi), bins_per_octave=fb_bpo),
	() -> auditory_fbank(stft))

# ╔═╡ 6a6510ca-6f35-4fcd-aebd-8eed21c931d0
isnothing(fb_msg) ? md"" : Markdown.MD(Markdown.Admonition("warning", "Parameters rejected — showing the default bank", [md"$(fb_msg)"]))

# ╔═╡ 5e3dda54-5f5f-4cea-a74f-971eb154d4c9
let W = get_data(fbank), f = get_freq(stft)
	plot(f, W'; legend=false, xguide="Hz", yguide="weight",
	     title="auditory_fbank: $(get_nbands(fbank)) bands, $(get_scale(fbank)), $(get_norm(fbank))", size=(720, 320))
	vline!(get_freq(fbank); c=:gray, alpha=0.3, ls=:dot)
end

# ╔═╡ b685b029-7275-4178-ab9f-15b9ce05f8e2
fbank

# ╔═╡ fb1ddbfa-82da-426a-ba59-2d34cc1c8fb5
md"""
### Where the centres go, scale by scale

The same 26 bands between 50 Hz and 7 kHz on every scale.
Mel, bark and ERB crowd the low frequencies where hearing is most selective; `logspace` and `octave` are geometric; `linspace` is uniform.
"""

# ╔═╡ a337fc50-9471-4b9e-a1ad-73c96e86ee7f
let p = plot(xguide="band", yguide="centre (Hz)", yscale=:log10, legend=:topleft, size=(720, 320))
	for (name, sc) in ("htk" => htk, "slaney" => slaney, "bark" => bark, "erb" => erb,
	                   "linspace" => linspace, "logspace" => logspace)
		fb = auditory_fbank(stft; nbands=26, scale=sc, freqrange=(50, 7000))
		plot!(p, get_freq(fb); label=name, marker=(:circle, 2))
	end
	fb = auditory_fbank(stft; nbands=24, scale=octave, freqrange=(50, 7000), bins_per_octave=4)
	plot!(p, get_freq(fb); label="octave (4 per octave)", marker=(:circle, 2))
end

# ╔═╡ bf326689-1a9e-4da5-bb46-d4b4875cfb89
md"""
### Gammatone (ERB) filterbank

nbands: $(@bind gt_nbands Slider(4:2:80; default=26, show_value=true))
norm: $(@bind gt_norm Select([bandwidth => "bandwidth", area => "area", none_norm => "none_norm"]))
low: $(@bind gt_lo Slider(0:10:2000; default=50, show_value=true)) Hz
high: $(@bind gt_hi Slider(1000:250:sr÷2; default=sr÷2, show_value=true)) Hz

Gammatone filters have long, asymmetric skirts instead of triangles: every band leaks a little into its neighbours, much like the cochlea.
"""

# ╔═╡ 1e11f001-cff4-4336-b8d3-3b732c1a8029
gtbank = gammatone_fbank(stft; nbands=gt_nbands, norm=gt_norm, freqrange=(gt_lo, gt_hi))

# ╔═╡ e507a31b-f644-4243-97c0-ecdaa635ab16
let W = get_data(gtbank), f = get_freq(stft)
	p1 = plot(f, W'; legend=false, xguide="Hz", yguide="weight", title="gammatone_fbank (linear)")
	p2 = plot(f, 20 .* log10.(max.(W', 1e-6)); legend=false, xguide="Hz", yguide="dB", ylims=(-80, 10), title="in dB", xscale=:identity)
	plot(p1, p2; layout=(1, 2), size=(720, 300))
end

# ╔═╡ cc2937c0-7146-42e7-8182-e2f93f5b2797
md"""
## 4. Linear spectrum — `LinSpec`

`LinSpec(front_end; freqrange, win_norm)` reproduces MATLAB's `linearSpectrum`: it keeps the bins inside `freqrange` and doubles every bin strictly between DC and Nyquist (the energy of the negative frequencies).
Its public data `get_data(lin)` is **frames × bins**.

| parameter | effect |
|:--|:--|
| `freqrange` | band of frequencies kept; everything outside is dropped (not zeroed), so the matrix shrinks. |
| `win_norm` | divide by `sum(w)²` (power) or `sum(w)` (magnitude): makes the level independent of the window length and shape, so a full-scale sine reads the same whatever `winsize` you use. |
"""

# ╔═╡ 4def5164-d52c-4976-890f-b97029af9a1e
md"""
low: $(@bind lin_lo Slider(0:50:4000; default=0, show_value=true)) Hz
high: $(@bind lin_hi Slider(500:250:sr÷2; default=sr÷2, show_value=true)) Hz
window normalisation $(@bind lin_winnorm CheckBox(default=false))
"""

# ╔═╡ 71ad6173-c9de-40f4-a252-e4019161ac6a
lin = LinSpec(stft; freqrange=(lin_lo, max(lin_hi, lin_lo + 100)), win_norm=lin_winnorm)

# ╔═╡ 4f99847e-3e1f-4d04-95e2-03509dd06668
plot(lin; top_db, freq_scale)

# ╔═╡ 80b15e1e-fff2-46d5-9e12-36397ba4a113
(size_frames_x_bins = size(get_data(lin)), freq_span = extrema(get_freq(lin)), max_value = maximum(get_spec(lin)))

# ╔═╡ a5654e33-af07-4e98-91c9-c652890b1152
md"""
## 5. Mel spectrogram — `MelSpec`

`MelSpec(front_end; win_norm, nbands, scale, norm, domain, style, bins_per_octave, freqrange)` designs a filterbank with `auditory_fbank` on the grid of the front end and applies it.
`MelSpec(front_end, fbank; win_norm)` applies a bank you designed yourself: here it is the bank from section 3, so **every filterbank control above also drives this spectrogram**.
Despite the historical name, `MelSpec` is the generic triangular-filterbank spectrogram and accepts every scale except `bark` (that is `BarkSpec`).
`get_data(mel)` is **frames × bands**.

| parameter | effect |
|:--|:--|
| `win_norm` | as in `LinSpec`; the keyword form defaults to `true`, the prebuilt-bank form to `false`. |
| filterbank keywords | see the table in section 3: fewer bands give a blurrier, more compact picture; a mel scale gives the low formants more rows. |
"""

# ╔═╡ 5c9bf6d8-125a-4be7-8b7d-bb953349c30a
md"window normalisation $(@bind mel_winnorm CheckBox(default=true))"

# ╔═╡ 72048626-d90c-4f76-a09c-989b393b8bad
mel = get_scale(fbank) === :bark ? BarkSpec(stft, fbank; win_norm=mel_winnorm) :
                                   MelSpec(stft, fbank; win_norm=mel_winnorm)

# ╔═╡ 54cf2d18-7653-4837-8fef-e623b8322b57
let
	p1 = plot(stft; top_db, title="Stft ($(get_nbins(stft)) bins)", freq_scale)
	p2 = plot(mel; top_db, title="$(nameof(typeof(mel))) ($(get_nbands(mel)) bands, $(get_scale(fbank)))", freq_scale)
	plot(p1, p2; layout=(2, 1), size=(720, 520))
end

# ╔═╡ 7dc58071-de07-41d8-811c-59ea932a3405
md"""
The keyword form gives the same result as the prebuilt bank when the arguments match:
"""

# ╔═╡ 064a238f-9b15-41bf-ae15-c6ae74baed48
let m2 = MelSpec(stft; nbands=26, scale=slaney, norm=area, freqrange=(50, 7000), win_norm=false)
	m1 = MelSpec(stft, auditory_fbank(stft; nbands=26, scale=slaney, norm=area, freqrange=(50, 7000)))
	(identical = get_data(m1) ≈ get_data(m2), size = size(get_data(m2)))
end

# ╔═╡ a177e7aa-9fa0-4784-aa3f-b78849ae251d
md"""
## 6. Bark spectrogram — `BarkSpec`

`BarkSpec(front_end; win_norm, nbands, norm, domain, style, freqrange)` is a `MelSpec` whose filterbank is always on the bark scale (MATLAB's `barkSpectrum`); it refuses a `scale` keyword.

nbands: $(@bind bark_nbands Slider(4:1:40; default=24, show_value=true))
norm: $(@bind bark_norm Select([bandwidth => "bandwidth", area => "area", none_norm => "none_norm"]))
domain: $(@bind bark_domain Select([:linear, :warped]))
low: $(@bind bark_lo Slider(0:10:1000; default=0, show_value=true)) Hz
high: $(@bind bark_hi Slider(1000:250:sr÷2; default=sr÷2, show_value=true)) Hz
window normalisation $(@bind bark_winnorm CheckBox(default=true))
"""

# ╔═╡ 26f0efc0-df29-4bb7-85ef-b6af2f2098e2
barkspec = BarkSpec(stft; nbands=bark_nbands, norm=bark_norm, domain=bark_domain,
                    freqrange=(bark_lo, bark_hi), win_norm=bark_winnorm)

# ╔═╡ 389e6280-5202-4863-a9f3-4534b54f40a0
let
	p1 = plot(barkspec; top_db, title="BarkSpec: $(get_nbands(barkspec)) bands")
	p2 = plot(get_freq(stft), get_data(get_fbank(barkspec))'; legend=false, xguide="Hz", title="its filters")
	plot(p1, p2; layout=(1, 2), size=(720, 300))
end

# ╔═╡ 55a15b5a-e098-4f6f-a132-bd93959268ae
md"""
## 7. ERB spectrogram — `ErbSpec`

`ErbSpec(front_end; win_norm, nbands, norm, freqrange)` multiplies the front end by a gammatone filterbank (MATLAB's `erbSpectrum`); `ErbSpec(front_end, fbank; win_norm)` takes a prebuilt `gammatone_fbank`.
Here it uses the gammatone bank designed in section 3.
Because gammatone skirts overlap, ERB spectrograms look smoother across bands than triangular ones.

window normalisation $(@bind erb_winnorm CheckBox(default=true))
"""

# ╔═╡ a9ada5c4-bc75-4baf-8a75-2305cd2f5a5b
erbspec = ErbSpec(stft, gtbank; win_norm=erb_winnorm)

# ╔═╡ 3c5e846b-6ce4-47f0-972b-2d38455c1418
plot(erbspec; top_db, title="ErbSpec: $(get_nbands(erbspec)) gammatone bands")

# ╔═╡ 7654aa9e-4ab3-44e0-81ca-2286c74d4410
md"""
## 8. Same stages on a wavelet front end

Every filterbank stage is written against the `AbstractSpectrogram` interface (`get_spec`, `get_freq`, …), not against `Stft`.
A `Cwt` scalogram sits on a **geometric** frequency grid, and `auditory_fbank(cwt; …)` simply evaluates the triangles on that grid.
The resulting `MelSpec` has exactly the same shape, frequencies and time axis as the STFT one.
The continuous wavelet transform itself is explored in notebook 5.

voices per octave: $(@bind cwt_voices Slider(4:2:24; default=8, show_value=true))
"""

# ╔═╡ cd0a6428-5a3a-46c2-a202-1cff159282ee
cwt = Cwt(frames; voices=cwt_voices, freqrange=(60, 7000))

# ╔═╡ 588287bb-a550-4a19-b022-8973bf9e17ef
begin
	mel_stft = MelSpec(stft; nbands=26, freqrange=(100, 7000))
	mel_cwt  = MelSpec(cwt;  nbands=26, freqrange=(100, 7000))
	(same_size = size(get_data(mel_stft)) == size(get_data(mel_cwt)),
	 same_centres = get_freq(mel_stft) ≈ get_freq(mel_cwt),
	 same_times = get_times(mel_stft) ≈ get_times(mel_cwt),
	 cwt_bins = get_nbins(cwt), stft_bins = get_nbins(stft))
end

# ╔═╡ 77cc0425-8a80-4aea-8836-149def981a20
let
	p1 = plot(cwt; top_db, freq_scale=:log10, title="Cwt ($(get_nbins(cwt)) scales)")
	p2 = plot(get_freq(cwt), get_data(auditory_fbank(cwt; nbands=26, freqrange=(100, 7000)))'; legend=false,
	          xscale=:log10, xguide="Hz", title="mel filters on the Cwt grid")
	p3 = plot(mel_stft; top_db, title="MelSpec of Stft")
	p4 = plot(mel_cwt; top_db, title="MelSpec of Cwt")
	plot(p1, p2, p3, p4; layout=(2, 2), size=(720, 560))
end

# ╔═╡ 076f48af-ecf7-428f-8eb7-e1a4c4b93f31
md"""
The same holds for `LinSpec`, `BarkSpec` and `ErbSpec`, and for every stage after them (cepstra, deltas, spectral descriptors).

## Next

* **3 — Cepstra**: MFCC and GTCC built on these spectrograms, deltas, and the MFCC recipes of MATLAB, HTK, Kaldi, librosa, ETSI and python_speech_features.
* **4 — Spectral descriptors**: centroid, spread, flux, rolloff and the audioFlux descriptor family.
* **5 — Time–frequency transforms**: CWT, CQT, S-transform, NSGT, synchrosqueezing, reassignment and Cohen-class distributions.

The full tour: 01 loading and frames · 02 STFT and filterbanks · 03 cepstra · 04 spectral descriptors · 05 time–frequency · 06 discrete wavelets and decompositions · 07 music and rhythm · 08 time domain and pitch · 09 signal processing and classic methods.
"""

# ╔═╡ Cell order:
# ╟─e49abb6e-7797-47c9-b6e1-4252128cc75b
# ╟─a7af9b36-a52f-43d4-b5e9-097373ee2cd1
# ╟─e5f1a706-fd08-442c-93a7-d624a331b438
# ╟─d8107959-f134-4ac4-841d-5df066bf0472
# ╠═d4cd84c8-c891-4880-8585-19796b4712f4
# ╠═5cb2027b-9430-422e-9561-7b17a5f13bc9
# ╟─627cd53e-692e-46a5-8bac-df8b6669577d
# ╟─f7d606db-c606-44e8-99d4-d9eefb6289a5
# ╠═c32f544a-7678-4532-b24f-fcaaeb9e0a8a
# ╠═825cfb11-636d-4ce9-baae-049fb0a31022
# ╠═cb046166-8b2e-4aad-9fb7-cb23f3dfe1bf
# ╟─e349e3e9-28d9-4c2f-bb4e-df8cfc1d7c48
# ╠═143061a1-2349-48f9-809a-c8c48d9010db
# ╠═9b36bfbe-1a58-4a1d-8783-5611752a4aff
# ╠═fef5910d-1f4d-43a0-becc-47fb46ed327c
# ╟─7349946f-7781-422d-852d-23c145b0e5c2
# ╠═b8de0ac7-8fed-48a0-82ae-61411b891754
# ╟─162e1fde-b3c0-4cab-bcca-6b96eb0edeef
# ╟─f533ad7e-52cb-4ab1-8bf5-cb27efea6eaf
# ╠═e5ee623a-54ec-4f95-8639-7b1d80a7c1d0
# ╠═4904640f-1df0-45a6-941d-2e7fd7eaa4c4
# ╠═33d0eb1e-8777-4d58-8f43-7cd2b8543348
# ╠═5194f139-0c89-45ac-bafc-b7d67916cb96
# ╟─a337406d-eb97-47fa-bd84-96bba3e90deb
# ╟─2e48dd93-3d60-426c-9c06-3bcc8e166701
# ╠═ec0fa127-1368-4ca3-91b4-135b664d273f
# ╠═55e01f2d-eb02-4647-aca8-eaec2420890e
# ╠═6a6510ca-6f35-4fcd-aebd-8eed21c931d0
# ╠═5e3dda54-5f5f-4cea-a74f-971eb154d4c9
# ╠═b685b029-7275-4178-ab9f-15b9ce05f8e2
# ╟─fb1ddbfa-82da-426a-ba59-2d34cc1c8fb5
# ╠═a337fc50-9471-4b9e-a1ad-73c96e86ee7f
# ╟─bf326689-1a9e-4da5-bb46-d4b4875cfb89
# ╠═1e11f001-cff4-4336-b8d3-3b732c1a8029
# ╠═e507a31b-f644-4243-97c0-ecdaa635ab16
# ╟─cc2937c0-7146-42e7-8182-e2f93f5b2797
# ╟─4def5164-d52c-4976-890f-b97029af9a1e
# ╠═71ad6173-c9de-40f4-a252-e4019161ac6a
# ╠═4f99847e-3e1f-4d04-95e2-03509dd06668
# ╠═80b15e1e-fff2-46d5-9e12-36397ba4a113
# ╟─a5654e33-af07-4e98-91c9-c652890b1152
# ╟─5c9bf6d8-125a-4be7-8b7d-bb953349c30a
# ╠═72048626-d90c-4f76-a09c-989b393b8bad
# ╠═54cf2d18-7653-4837-8fef-e623b8322b57
# ╟─7dc58071-de07-41d8-811c-59ea932a3405
# ╠═064a238f-9b15-41bf-ae15-c6ae74baed48
# ╟─a177e7aa-9fa0-4784-aa3f-b78849ae251d
# ╠═26f0efc0-df29-4bb7-85ef-b6af2f2098e2
# ╠═389e6280-5202-4863-a9f3-4534b54f40a0
# ╟─55a15b5a-e098-4f6f-a132-bd93959268ae
# ╠═a9ada5c4-bc75-4baf-8a75-2305cd2f5a5b
# ╠═3c5e846b-6ce4-47f0-972b-2d38455c1418
# ╟─7654aa9e-4ab3-44e0-81ca-2286c74d4410
# ╠═cd0a6428-5a3a-46c2-a202-1cff159282ee
# ╠═588287bb-a550-4a19-b022-8973bf9e17ef
# ╠═77cc0425-8a80-4aea-8836-149def981a20
# ╟─076f48af-ecf7-428f-8eb7-e1a4c4b93f31
