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

# ╔═╡ 5abedcfe-4595-4868-b592-c32101b87179
begin
	import Pkg
	# the notebook environment next to this file pins Audio911 (from this repository), PlutoUI and Plots
	Pkg.activate(@__DIR__)
	Pkg.instantiate()
	using Audio911, PlutoUI, Plots
	include(joinpath(@__DIR__, "common.jl"))
	gr(); default(size=(720, 300), legend=:topright, titlefontsize=10, guidefontsize=9)
end

# ╔═╡ d5cd1c3f-2814-42e6-a1ae-e8b5519cc537
md"""
# Audio911 · 1 — Loading audio and framing

This is the first notebook of the Audio911 interactive tour.
Every Audio911 pipeline has the same shape:

```
AudioFile → Frames → front end → filterbank → cepstrum / feature
```

This notebook covers the first two stages: **loading** a signal and cutting it into **frames**.
Every other notebook builds on the objects shown here, so the parameters you learn below reappear everywhere.

Move a slider or change a menu and every cell that depends on it re-runs.
"""

# ╔═╡ a93c465a-9e82-4603-898f-b01f57ccd4d4
TableOfContents()

# ╔═╡ e1628450-7f7e-4668-97fa-8ac54c03a915
md"""
## 1. Loading

`load(path; sr, format, norm)` reads WAV, FLAC, OGG and MP3 through libsndfile and mpg123 and returns an `AudioFile`.
`AudioFile(x, sr)` wraps samples you already hold.

| parameter | effect |
|:--|:--|
| `sr` | target sample rate. `0` keeps the file's rate; any other value resamples. A lower rate makes every later stage cheaper but discards everything above `sr/2`. |
| `format` | element type, `Float32` (the library default, used end to end) or `Float64`. |
| `norm` | peak-normalise to ±1. Useful when comparing files recorded at different levels. |

The menu below offers the speech file shipped with the tests and three synthetic signals chosen to make resolution effects visible.
"""

# ╔═╡ cf96ca23-78cc-4a28-b3c3-6c37405f1fe0
md"""
Signal: $(@bind signal_name Select(SIGNALS))
Sample rate: $(@bind sr Select([16000 => "16 kHz", 8000 => "8 kHz", 22050 => "22.05 kHz"]))
"""

# ╔═╡ 2be80196-f155-4679-b3d2-656eb6af739d
audio = let a = demo_signal(signal_name)
	# resample by reloading through the file loader for speech, or re-synthesising for the others
	signal_name == "speech" ?
		load(joinpath(SAMPLES_DIR, "test.wav"); sr, format=Float32, norm=true) :
		demo_signal(signal_name; sr)
end

# ╔═╡ 1ee2d15c-84a8-4a7e-82fc-5de6e3b38ff2
plot(audio)

# ╔═╡ b5daa6ed-bc35-4fa9-a50a-02159af08768
audio_player(audio)

# ╔═╡ 20544789-7632-4dcf-9281-4bd3882ae4ca
md"""
Try switching the speech file to **8 kHz**: it still sounds intelligible, but consonants such as *s* lose their brightness because all content above 4 kHz is gone.
The same file at 22.05 kHz is upsampled; no new information appears.

The file formats themselves are detected from the extension (and the header for WAV):
"""

# ╔═╡ 5cd8b5f6-0bf8-4b51-808d-36a07582ef28
[(f, detect_format(joinpath(SAMPLES_DIR, f))) for f in ("test.wav", "test.flac", "test.ogg", "test.mp3")]

# ╔═╡ d1080358-29db-461e-92cb-ec15a95b278e
md"""
## 2. Frames

`Frames(audio; kwargs...)` cuts the signal into overlapping frames and attaches an analysis window.
Frames are lazy: nothing is copied until a transform streams through them.

| parameter | effect |
|:--|:--|
| `winsize` | frame length in samples. Longer frames resolve closer frequencies (resolution `sr / winsize` Hz) but blur fast events in time. |
| `winstep` | hop between frames. A smaller hop gives more frames per second (a smoother time axis) at proportionally more cost; it does not improve frequency resolution. |
| `type` | window function. It trades main-lobe width (how close two frequencies can be) against side-lobe level (how far a loud component leaks). |
| `periodic` | `true` gives the DFT-even window used by MATLAB's `"periodic"` option; `false` the symmetric window used for filter design. |
| `center` | pad `winsize ÷ 2` on both sides so frame `i` is centred on sample `(i-1)·winstep` (librosa's `center=True`). |
| `pad_mode` | how the centre padding is filled: `:constant` (zeros), `:reflect` or `:edge`. |
| `preemph` | per-frame pre-emphasis coefficient (HTK / Kaldi). `0.97` boosts high frequencies by about 6 dB/octave. |
| `dc_removal` | subtract each frame's mean (Kaldi's `remove_dc_offset`). |
| `pad_end` | zero-pad the end so a trailing partial segment still yields a frame (python_speech_features). |
"""

# ╔═╡ 2814e367-2474-4e13-8e1c-dd03dab50d2e
md"""
winsize: $(@bind winsize Select([128, 256, 400, 512, 1024, 2048]; default=512))
hop (fraction of winsize): $(@bind hopfrac Select([1.0 => "1 (no overlap)", 0.5 => "1/2", 0.25 => "1/4", 0.125 => "1/8"]; default=0.5))

window: $(@bind wintype Select([hanning => "hanning", hamming => "hamming", blackman => "blackman", rect => "rect", triang => "triang", bartlett => "bartlett", bartlett_hann => "bartlett_hann", cosine => "cosine", lanczos => "lanczos", povey => "povey"]))
periodic $(@bind periodic CheckBox(default=true))
center $(@bind center CheckBox(default=false))
pad mode $(@bind pad_mode Select([:constant, :reflect, :edge]))

pre-emphasis: $(@bind preemph Slider(0:0.01:0.99; default=0, show_value=true))
DC removal $(@bind dc_removal CheckBox(default=false))
pad end $(@bind pad_end CheckBox(default=false))
"""

# ╔═╡ 4a32a9bf-1cfa-47fe-9f1c-2eb9128d80ed
frames = Frames(audio; winsize, winstep=max(1, round(Int, winsize * hopfrac)),
                type=wintype, periodic, center, pad_mode, preemph, dc_removal, pad_end)

# ╔═╡ c71f79b7-f9e3-45b6-aa2f-49c6ea3b04bb
plot(frames)

# ╔═╡ 90e71e1b-7136-4627-9c03-774045190bba
md"""
The dashed curve is the window placed on the first frame, so the frame length can be read directly off the time axis.
Some numbers derived from these choices:
"""

# ╔═╡ d391996c-2b42-4175-be27-b6ce918dcff5
(frames_count = length(frames),
 frame_ms = 1000 * winsize / sr,
 hop_ms = 1000 * get_step(frames) / sr,
 frames_per_second = sr / get_step(frames),
 bin_spacing_Hz = sr / winsize,
 frame_matrix = size(get_data(frames)))

# ╔═╡ d6ecc63c-ffbd-4b20-b51f-fa0bae27d229
md"""
### What one frame looks like

Pick a frame and see the raw segment next to the windowed segment the FFT will actually see.
With pre-emphasis on, the windowed frame looks "rougher": the filter `y[n] = x[n] - α·x[n-1]` emphasises rapid changes.
"""

# ╔═╡ 6bed5cd5-551b-4303-9b7c-0fed380ebd0c
@bind frame_idx Slider(1:length(frames); default=cld(length(frames), 2), show_value=true)

# ╔═╡ 93e72a79-3574-4b23-a0ca-0af8d12131d0
let X = get_data(frames), w = get_window(frames)
	# get_data(frames) holds each frame after DC removal and pre-emphasis, before windowing
	seg = X[:, frame_idx]
	p1 = plot(seg; label="frame $frame_idx", title="raw segment (after DC removal / pre-emphasis)")
	p2 = plot(seg .* w; label="windowed", title="what the FFT sees", c=2)
	plot(p1, p2; layout=(2, 1), size=(720, 420), xguide="sample")
end

# ╔═╡ b83f6e6d-2027-4c27-83e0-aee143089146
md"""
### Window shapes and their spectra

A window's spectrum is the "blur kernel" every spectral line gets convolved with.
The main lobe width decides how close two tones can be before they merge; the side lobes decide how much a loud tone leaks over a quiet one.
`rect` has the narrowest main lobe but side lobes only 13 dB down; `blackman` has a wide main lobe but side lobes below −58 dB.
"""

# ╔═╡ 25293b0c-bf09-4388-a71b-5641fe0ce546
let n = 256, nfft = 8192
	wins = [rect, hanning, hamming, blackman, povey]
	p1 = plot(title="time domain", xguide="sample")
	p2 = plot(title="spectrum (dB)", xguide="bins from centre (× n/nfft)", ylims=(-120, 5))
	for f in wins
		w = f === povey ? povey(n) : f(n)
		plot!(p1, w; label=string(f))
		W = abs.(Audio911.FFTW.fft(vcat(w, zeros(nfft - n))))
		W = 20 .* log10.(max.(W ./ maximum(W), 1e-12))
		k = -nfft÷32:nfft÷32
		plot!(p2, k .* n ./ nfft, circshift(W, nfft÷32)[1:length(k)]; label=string(f))
	end
	plot(p1, p2; layout=(1, 2), size=(720, 300))
end

# ╔═╡ 01a938e0-b10e-4a20-9384-c8d143f4138f
md"""
## 3. Pre-emphasis and de-emphasis on the whole signal

`preemphasis(x; coef)` applies `y[n] = x[n] - coef·x[n-1]` to the whole signal; `deemphasis(y; coef)` is its exact inverse.
Speech energy falls with frequency, so pre-emphasis flattens its spectrum and gives the upper formants a fair share of the dynamic range before a filterbank.
"""

# ╔═╡ 4f427ce4-b744-4a5c-88a3-3134ae6c1309
@bind coef Slider(0:0.01:0.99; default=0.97, show_value=true)

# ╔═╡ f6d64778-fdea-48f6-b00f-11e66eaf6444
begin
	x0 = vec(get_data(audio))
	xp = preemphasis(x0; coef)
	xr = deemphasis(xp; coef)
	(max_reconstruction_error = maximum(abs, xr .- x0),)
end

# ╔═╡ 2ca334a2-bd9e-44b3-b357-ca38406d6df0
let
	spec(x) = let X = abs.(Audio911.FFTW.rfft(x)); 20 .* log10.(max.(X ./ maximum(X), 1e-8)) end
	f = range(0, sr / 2; length=length(x0) ÷ 2 + 1)
	plot(f, spec(x0); label="original", alpha=0.6, xguide="Hz", yguide="dB", title="long-term spectrum", ylims=(-100, 5))
	plot!(f, spec(xp); label="pre-emphasised (coef=$coef)", alpha=0.6)
end

# ╔═╡ ed150e36-80cb-4dee-81e2-d7a18f8fa17f
audio_player(xp, sr)

# ╔═╡ ee614ef9-f069-4dd5-98e9-9f0c49598e99
md"""
## Next

* **2 — STFT and filterbanks**: from frames to a spectrogram, then to linear, mel, bark and ERB bands.
* **3 — Cepstra**: MFCC, GTCC, deltas and the MFCC variants of MATLAB, HTK, Kaldi, librosa, ETSI and python_speech_features.

All notebooks live next to this one in the `notebooks/` folder; see `notebooks/README.md` for the full list.
"""

# ╔═╡ Cell order:
# ╟─d5cd1c3f-2814-42e6-a1ae-e8b5519cc537
# ╟─5abedcfe-4595-4868-b592-c32101b87179
# ╟─a93c465a-9e82-4603-898f-b01f57ccd4d4
# ╟─e1628450-7f7e-4668-97fa-8ac54c03a915
# ╟─cf96ca23-78cc-4a28-b3c3-6c37405f1fe0
# ╠═2be80196-f155-4679-b3d2-656eb6af739d
# ╠═1ee2d15c-84a8-4a7e-82fc-5de6e3b38ff2
# ╠═b5daa6ed-bc35-4fa9-a50a-02159af08768
# ╟─20544789-7632-4dcf-9281-4bd3882ae4ca
# ╠═5cd8b5f6-0bf8-4b51-808d-36a07582ef28
# ╟─d1080358-29db-461e-92cb-ec15a95b278e
# ╟─2814e367-2474-4e13-8e1c-dd03dab50d2e
# ╠═4a32a9bf-1cfa-47fe-9f1c-2eb9128d80ed
# ╠═c71f79b7-f9e3-45b6-aa2f-49c6ea3b04bb
# ╟─90e71e1b-7136-4627-9c03-774045190bba
# ╠═d391996c-2b42-4175-be27-b6ce918dcff5
# ╟─d6ecc63c-ffbd-4b20-b51f-fa0bae27d229
# ╠═6bed5cd5-551b-4303-9b7c-0fed380ebd0c
# ╠═93e72a79-3574-4b23-a0ca-0af8d12131d0
# ╟─b83f6e6d-2027-4c27-83e0-aee143089146
# ╠═25293b0c-bf09-4388-a71b-5641fe0ce546
# ╟─01a938e0-b10e-4a20-9384-c8d143f4138f
# ╠═4f427ce4-b744-4a5c-88a3-3134ae6c1309
# ╠═f6d64778-fdea-48f6-b00f-11e66eaf6444
# ╠═2ca334a2-bd9e-44b3-b357-ca38406d6df0
# ╠═ed150e36-80cb-4dee-81e2-d7a18f8fa17f
# ╟─ee614ef9-f069-4dd5-98e9-9f0c49598e99
