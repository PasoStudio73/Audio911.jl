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

# ╔═╡ 9a6d4816-d46c-4865-b80b-d85676ffcf0f
begin
	import Pkg
	# the notebook environment next to this file pins Audio911 (from this repository), PlutoUI and Plots
	Pkg.activate(@__DIR__)
	Pkg.instantiate()
	using Audio911, PlutoUI, Plots
	include(joinpath(@__DIR__, "common.jl"))
	gr(); default(size=(720, 300), legend=:topright, titlefontsize=10, guidefontsize=9)
end

# ╔═╡ f3474cc5-d683-4319-92ae-c8686cde6028
md"""
# Audio911 · 6 — Discrete wavelets and adaptive decompositions

The continuous wavelet transform of notebook 5 samples scale finely and is highly redundant.
This notebook covers the **discrete** family, which splits a signal into octave (or equal-width) subbands with critically sampled or undecimated coefficients, and three **adaptive** decompositions whose bands are chosen by the signal itself:

| algorithm | what it returns | bands |
|:--|:--|:--|
| `dwt`, `Dwt` | Mallat cascade: approximation + one detail per level | octaves, fixed |
| `wpt`, `Wpt` | full wave-packet tree | `2^level` equal bands, fixed |
| `swt`, `Swt` | undecimated (à trous) cascade | octaves, fixed, shift invariant |
| `emd` | intrinsic mode functions by sifting | adaptive, data driven |
| `Hht` | Hilbert-Huang spectrum of the IMFs | adaptive, instantaneous frequency |
| `ewt`, `Ewt` | empirical wavelet filter bank | adaptive, from spectral peaks |

The lowercase functions work on a whole signal; the capitalised front ends pool the subbands on a `Frames` grid, so they plug into the rest of the pipeline like `Stft` does.
"""

# ╔═╡ 5596a305-1317-4e36-861b-93d81fc0de9d
TableOfContents()

# ╔═╡ 15d129cf-98ec-4a6e-b2d0-606c55c73e16
md"""
## 0. The signal

Signal: $(@bind signal_name Select(SIGNALS))
Sample rate: $(@bind sr Select([16000 => "16 kHz", 8000 => "8 kHz"]))
"""

# ╔═╡ dce70742-103f-4921-8724-9498c7fa8022
audio = signal_name == "speech" ?
	load(joinpath(SAMPLES_DIR, "test.wav"); sr, format=Float32, norm=true) :
	demo_signal(signal_name; sr)

# ╔═╡ fcc6a839-f88b-46ad-9680-f2fdc2197f8a
x = vec(get_data(audio)); audio_player(audio)

# ╔═╡ 99d66a33-4b6d-4c26-9c95-641a9ad974d4
md"""
## 1. The 51 discrete wavelets

`wavelet_filters(name)` returns the four filters `(loD, hiD, loR, hiR)`: low- and high-pass for **decomposition** and for **reconstruction**.
`DISCRETE_WAVELETS` lists every accepted name (audioFlux's tables).

| family | names | properties |
|:--|:--|:--|
| Haar | `haar` (= `db1`) | 2 taps, perfect time localisation, poor frequency selectivity |
| Daubechies | `db2`–`db10`, `db20`, `db30`, `db40` | orthogonal, `dbN` has `2N` taps and `N` vanishing moments; very asymmetric |
| Symlets | `sym2`–`sym10`, `sym20`, `sym30` | orthogonal, as `dbN` but as symmetric as possible (near-linear phase) |
| Coiflets | `coif1`–`coif5` | orthogonal, `6N` taps, vanishing moments on the scaling function too |
| Fejér-Korovkin | `fk4`–`fk22` | orthogonal, optimised for sharp frequency separation |
| Biorthogonal | `bior1.1`–`bior6.8` | symmetric (linear phase); analysis and synthesis filters differ |
| Discrete Meyer | `dmey` | 102 taps approximating the band-limited Meyer wavelet; the most frequency-selective |

Longer filters give flatter pass-bands and steeper transitions between subbands, at the cost of worse time localisation and more computation.
"""

# ╔═╡ 5618323a-be51-434a-ad1a-39d36e4b185d
md"""
Wavelet: $(@bind fwave Select(collect(DISCRETE_WAVELETS); default="db4"))
"""

# ╔═╡ 6057b0fb-427b-4cb7-8a6d-8201dda7f9c8
let (loD, hiD, loR, hiR) = wavelet_filters(fwave)
	st(v, t) = plot(0:length(v)-1, v; line=:stem, marker=:circle, ms=2, label="", title=t)
	p = plot(st(loD, "loD (analysis low-pass)"), st(hiD, "hiD (analysis high-pass)"),
	         st(loR, "loR (synthesis low-pass)"), st(hiR, "hiR (synthesis high-pass)"); layout=(2, 2))
	nfft = 1024
	H(v) = abs.(Audio911.FFTW.rfft(vcat(v, zeros(nfft - length(v)))))
	ω = range(0, 1; length=nfft ÷ 2 + 1)
	q = plot(ω, H(loD); label="|LoD|", xguide="normalised frequency (× Nyquist)", title="frequency responses: $fwave, $(length(loD)) taps")
	plot!(q, ω, H(hiD); label="|HiD|")
	plot(p, q; layout=@layout([a{0.6h}; b]), size=(720, 560))
end

# ╔═╡ 79f51240-4732-46b4-baee-93a4172c3d10
md"""
## 2. Discrete wavelet transform — `dwt`

`dwt(x; wavelet="sym4", level=log2(length(x)) - 1) -> (coefs, image)`

Each level low-pass and high-pass filters the previous approximation and keeps every other sample, so level `j` holds the detail band `[sr/2^(j+1), sr/2^j]` with `N/2^j` coefficients.

| parameter | effect |
|:--|:--|
| `wavelet` | the filter pair (section 1). It changes how cleanly energy is separated between neighbouring octaves. |
| `level` | number of analysis steps. Each extra level halves the approximation band and adds one coarser detail band. The signal length must be a multiple of `2^level` (we zero-pad here). |

`coefs` is `[cA_level; cD_level; …; cD_1]` (PyWavelets' `wavedec` order, `length(x)` values in total, critically sampled).
`image` is audioFlux's `level × N` matrix, each coefficient repeated over the samples it covers.
"""

# ╔═╡ a833717e-67e3-438a-949d-9c2580a9f08b
md"""
Wavelet: $(@bind dwave Select(collect(DISCRETE_WAVELETS); default="sym4"))
Level: $(@bind dlevel Slider(1:10; default=5, show_value=true))
"""

# ╔═╡ 1779345e-2060-4d5e-82b2-0c36eaff2a94
begin
	# zero-pad to a multiple of 2^level, as the Dwt front end does
	pad2(v, level) = let m = 1 << level; vcat(v, zeros(eltype(v), cld(length(v), m) * m - length(v))) end
	xd = pad2(x, dlevel)
	coefs, img = dwt(xd; wavelet=dwave, level=dlevel)
	# split coefs into [cA_L, cD_L, ..., cD_1]
	function split_bands(c, level)
		N = length(c); lo = N >> level
		bands = [c[1:lo]]
		for lev in level:-1:1
			len = N >> lev
			push!(bands, c[lo+1:lo+len]); lo += len
		end
		bands
	end
	bands = split_bands(coefs, dlevel)
	band_names = vcat("cA$dlevel", ["cD$l" for l in dlevel:-1:1])
	band_ranges = vcat((0.0, sr / 2^(dlevel + 1)), [(sr / 2^(l + 1), sr / 2^l) for l in dlevel:-1:1])
end;

# ╔═╡ eb52c141-08ed-49c8-8cf5-1a9e1e879443
let N = length(xd)
	ps = [plot(range(0, N / sr; length=length(b)), b; label="", title="$(n)  $(round(Int, r[1]))–$(round(Int, r[2])) Hz, $(length(b)) coefs",
	           titlefontsize=8, ticks=nothing) for (n, b, r) in zip(band_names, bands, band_ranges)]
	plot(ps...; layout=(length(ps), 1), size=(720, 90 * length(ps)), xticks=:auto)
end

# ╔═╡ 711e7075-2352-44eb-a66d-59fafe070332
md"""
Energy share per band (the approximation plus all details add up to the signal energy for orthogonal wavelets):
"""

# ╔═╡ b07579eb-dbfe-43b3-8b4a-b2177ca4dfa0
let e = [sum(abs2, b) for b in bands]
	bar(band_names, 100 .* e ./ sum(e); label="", ylabel="% of energy", title="energy per band ($dwave, level $dlevel)")
end

# ╔═╡ 4097dc44-8203-486c-adc0-291410bfdea3
heatmap(range(0, length(xd) / sr; length=size(img, 2)), 1:size(img, 1), abs.(img);
        xguide="Time (s)", yguide="image row (1 = coarsest detail)", title="audioFlux image (|coef|)", c=:viridis)

# ╔═╡ c94f5d5c-6c53-4919-8bd8-06c9765e08ab
md"""
## 3. Reconstruction, band isolation and denoising

Audio911 provides the analysis side only.
Because the transform is linear, the inverse is a few lines: each synthesis step upsamples and filters with the reconstruction pair `(loR, hiR)`.
The helper below undoes `dwt` exactly (to float precision) for all 51 wavelets, including the biorthogonal ones.
"""

# ╔═╡ dde968bb-4d2e-4ea6-a52a-b00f2f1ff30b
begin
	function idwt_step(c::AbstractVector{T}, d::AbstractVector, lo, hi) where T
		n2 = length(c); n = 2n2; L = length(lo); h = L ÷ 2
		a = zeros(T, n)
		for i in 1:n2, j in 1:L
			k = mod1(2i + L - j - h, n)
			a[k] += T(lo[j]) * c[i] + T(hi[j]) * d[i]
		end
		a
	end
	"inverse of `dwt(x; wavelet, level)` from its `coefs` vector"
	function idwt(c::AbstractVector, level; wavelet)
		_, _, loR, hiR = wavelet_filters(wavelet)
		lo, hi = reverse(loR), reverse(hiR)
		N = length(c); n = N >> level
		a = c[1:n]
		while n < N
			a = idwt_step(a, c[n+1:2n], lo, hi); n *= 2
		end
		a
	end
	(reconstruction_error = maximum(abs, idwt(coefs, dlevel; wavelet=dwave) .- xd),)
end

# ╔═╡ 780f73b7-29b0-4a35-aff4-4fa41a671b11
md"""
### Keep only some bands

Zero the coefficients of the unticked bands and invert: an octave-band equaliser in one line.
Keeping only `cA` gives a low-passed signal, keeping only `cD1` the top octave (the hiss and sibilants of speech).
"""

# ╔═╡ 72c19158-939e-4f04-a6c2-016168dd5f9b
@bind keep_bands MultiCheckBox(band_names; default=band_names[1:2])

# ╔═╡ 1493693e-7d3c-4520-9696-319efc54886d
begin
	masked = reduce(vcat, [n in keep_bands ? b : zero(b) for (n, b) in zip(band_names, bands)])
	xband = idwt(masked, dlevel; wavelet=dwave)
	plot(range(0, length(xd) / sr; length=length(xd)), [xd xband]; label=["original" "kept bands"], alpha=[0.4 1], xguide="Time (s)")
end

# ╔═╡ aaa6b09a-2baf-42d9-ab44-55679e35b9d9
audio_player(xband, sr)

# ╔═╡ 7ac6903e-9999-43d4-a224-b317548f3a20
md"""
### Wavelet shrinkage denoising

White noise spreads evenly over all coefficients, while a signal concentrates in a few large ones.
Thresholding the detail coefficients therefore removes noise and keeps the signal (Donoho & Johnstone).
The noise level `σ` is estimated from the finest detail as `median(|cD1|) / 0.6745`, and the universal threshold is `σ·√(2 ln N)`, scaled by the multiplier below.

| control | effect |
|:--|:--|
| SNR | level of the white noise added to the clean signal |
| threshold × | multiplier on the universal threshold: larger removes more noise and more signal detail |
| rule | `hard` zeroes small coefficients and keeps the rest; `soft` also shrinks the survivors by the threshold (smoother, slightly duller) |
| wavelet / level | from section 2; more levels let the shrinkage reach lower frequencies |
"""

# ╔═╡ edf3f78e-8472-4726-b5b0-aab74f6c9131
md"""
SNR (dB): $(@bind snr Slider(-5:1:30; default=10, show_value=true))
threshold ×: $(@bind thr_mult Slider(0:0.05:2; default=1, show_value=true))
rule: $(@bind thr_rule Select(["soft", "hard"]))
"""

# ╔═╡ 78191453-da25-4d5e-90d1-39be7be29690
begin
	noisy = let n = randn(Float32, length(xd))
		n .*= sqrt(sum(abs2, xd) / sum(abs2, n)) * 10f0^(-snr / 20)
		xd .+ n
	end
	cn, _ = dwt(noisy; wavelet=dwave, level=dlevel)
	nA = length(cn) >> dlevel
	σ = Audio911.median(abs.(cn[end - length(cn) ÷ 2 + 1:end])) / 0.6745
	λ = thr_mult * σ * sqrt(2log(length(cn)))
	shrink(c) = thr_rule == "hard" ? (abs(c) > λ ? c : zero(c)) : sign(c) * max(abs(c) - λ, 0)
	cden = vcat(cn[1:nA], shrink.(cn[nA+1:end]))
	denoised = idwt(cden, dlevel; wavelet=dwave)
	snr_db(ref, y) = 10log10(sum(abs2, ref) / sum(abs2, y .- ref))
	(input_SNR_dB = round(snr_db(xd, noisy); digits=2), output_SNR_dB = round(snr_db(xd, denoised); digits=2),
	 threshold = λ, zeroed_details_pct = round(100 * count(iszero, cden[nA+1:end]) / (length(cden) - nA); digits=1))
end

# ╔═╡ ec4a1820-04dd-4f09-a324-a2d7d330c510
let t = range(0, length(xd) / sr; length=length(xd))
	plot(t, noisy; label="noisy", alpha=0.35, xguide="Time (s)")
	plot!(t, denoised; label="denoised ($thr_rule)")
end

# ╔═╡ dddd2a5a-0bc3-46fb-adf0-9cf98b75fbc5
md"noisy: $(audio_player(noisy, sr)) denoised: $(audio_player(denoised, sr))"

# ╔═╡ 91627684-cffe-4866-8230-1964f5e2f0f3
md"""
## 4. Wave-packet transform — `wpt`

`wpt(x; wavelet="sym4", level=log2(length(x)) - 1) -> (coefs, image)`

The DWT only splits the low-pass branch again; the wave-packet transform splits **both** branches at every level.
After `level` steps there are `2^level` leaves of equal width `sr / 2^(level+1)`, reordered into frequency order (the high-pass children are swapped, the Gray-code fix).

| parameter | effect |
|:--|:--|
| `wavelet` | as for `dwt`; long filters matter more here because every band edge is crossed many times |
| `level` | `2^level` bands: each extra level doubles the frequency resolution and halves the time resolution of every band |

`image` is `2^level × N`, one row per band.
"""

# ╔═╡ 812f1faa-b305-42e6-8985-8b7d2b528a0d
md"""
Wavelet: $(@bind wwave Select(collect(DISCRETE_WAVELETS); default="sym8"))
Level: $(@bind wlevel Slider(1:8; default=5, show_value=true))
"""

# ╔═╡ a3076844-0223-46c4-ac0d-01870f545f6b
let xw = pad2(x, wlevel)
	_, wimg = wpt(xw; wavelet=wwave, level=wlevel)
	bw = sr / 2 / (1 << wlevel)
	# time axis decimated so the heatmap stays light
	step = max(1, size(wimg, 2) ÷ 1500)
	cols = 1:step:size(wimg, 2)
	z = 20 .* log10.(abs.(wimg[:, cols]) .+ 1f-6)
	heatmap((cols .- 1) ./ sr, ((1:size(wimg, 1)) .- 0.5) .* bw, z; clims=(maximum(z) - 70, maximum(z)),
	        xguide="Time (s)", yguide="Frequency (Hz)", colorbar_title="dB",
	        title="wpt: $(1 << wlevel) bands of $(round(bw; digits=1)) Hz ($wwave)")
end

# ╔═╡ 05b413f5-6380-45f1-bc6a-00c37e7ccaf7
md"""
With the **two close tones** signal, raise the level until 440 and 470 Hz fall in different bands (level ≥ 8 at 16 kHz gives 31 Hz bands); with the **chirp**, low levels show a staircase and high levels a smooth, but temporally blurred, line.
"""

# ╔═╡ eeeaf1ba-a03c-419d-887e-f36eae9529ef
md"""
## 5. Stationary wavelet transform — `swt`

`swt(x; wavelet="sym4", level=1) -> (A, D)`

The à trous algorithm skips the decimation and instead upsamples the filters by `2^(i-1)` at level `i`.
Every level keeps all `N` samples (`A` and `D` are `level × N`): the transform is `level + 1` times redundant but **shift invariant**, which is why it is preferred for detection and denoising.

| parameter | effect |
|:--|:--|
| `wavelet` | as for `dwt` |
| `level` | number of octaves; note the default is `1` here, unlike `dwt` |
"""

# ╔═╡ 102e0047-cf7e-4578-bc69-a0130b7b54e6
md"""
Wavelet: $(@bind swave Select(collect(DISCRETE_WAVELETS); default="db4"))
Level: $(@bind slevel Slider(1:8; default=4, show_value=true))
"""

# ╔═╡ 7b2440f6-4b80-4a59-be1b-7a01a0f4b915
begin
	xs = pad2(x, slevel)
	A, D = swt(xs; wavelet=swave, level=slevel)
	let t = (0:length(xs)-1) ./ sr
		ps = [plot(t, D[i, :]; label="", title="D$i  $(round(Int, sr / 2^(i+1)))–$(round(Int, sr / 2^i)) Hz", titlefontsize=8) for i in 1:slevel]
		push!(ps, plot(t, A[slevel, :]; label="", title="A$slevel  0–$(round(Int, sr / 2^(slevel+1))) Hz", titlefontsize=8, c=2))
		plot(ps...; layout=(length(ps), 1), size=(720, 90 * length(ps)))
	end
end

# ╔═╡ 9cff018e-1d32-419e-b13a-1636194f4853
md"""
### Shift invariance

Circularly shift the signal by a few samples and compare the energy in each detail band before and after.
The DWT's decimation makes its band energies wobble with the shift; the SWT's stay put (up to the wrap-around).

Shift (samples): $(@bind shift Slider(0:1:31; default=1, show_value=true))
"""

# ╔═╡ 26028605-2977-4d61-b31b-b0f8a51af9c2
let y = circshift(xs, shift)
	cx, _ = dwt(xs; wavelet=swave, level=slevel); cy, _ = dwt(y; wavelet=swave, level=slevel)
	ex = [sum(abs2, b) for b in split_bands(cx, slevel)[2:end]]
	ey = [sum(abs2, b) for b in split_bands(cy, slevel)[2:end]]
	_, Dx = swt(xs; wavelet=swave, level=slevel); _, Dy = swt(y; wavelet=swave, level=slevel)
	sx = [sum(abs2, Dx[i, :]) for i in slevel:-1:1]; sy = [sum(abs2, Dy[i, :]) for i in slevel:-1:1]
	rel(a, b) = 100 .* abs.(b .- a) ./ max.(a, eps(Float32))
	lab = ["D$l" for l in slevel:-1:1]
	plot(lab, [rel(ex, ey) rel(sx, sy)]; label=["dwt" "swt"], marker=:circle, lw=2,
	     ylabel="% change of band energy", title="effect of a $shift-sample shift")
end

# ╔═╡ 7fe6910d-a9a7-490f-b332-bec537c7a29f
md"""
## 6. Frame-pooled front ends — `Dwt`, `Wpt`, `Swt`

`Dwt(frames; wavelet="sym4", level=5, spectrum=power)` (and `Wpt`, `Swt` with the same keywords) zero-pad the framed signal to a multiple of `2^level`, run the transform, hold every coefficient over the samples it covers and average its power (or magnitude) over each frame with the frame window.
The result is a `DiscreteWavelet <: AbstractSpectrogram`: `get_spec` is `bands × frames` in ascending frequency and `get_freq` gives the band centres, so filterbanks and spectral descriptors run on it unchanged.

| parameter | effect |
|:--|:--|
| `wavelet`, `level` | as for the whole-signal transforms: `Dwt`/`Swt` give `level + 1` octave bands, `Wpt` gives `2^level` equal bands |
| `spectrum` | `power` (squared coefficients) or `magnitude` |
| `winsize`, `winstep`, `type` (via `Frames`) | only the pooling grid: larger frames average more coefficients per column (smoother, coarser time axis). Called with an `AudioFile` the window defaults to `rect`. |
"""

# ╔═╡ 7bcac67d-1a62-43ab-bbf1-316b3a36b5d8
md"""
Front end: $(@bind fe_kind Select(["Dwt", "Wpt", "Swt"]))
wavelet: $(@bind fe_wave Select(collect(DISCRETE_WAVELETS); default="sym4"))
level: $(@bind fe_level Slider(1:8; default=5, show_value=true))
spectrum: $(@bind fe_spec Select([power => "power", magnitude => "magnitude"]))

winsize: $(@bind fe_winsize Select([128, 256, 512, 1024]; default=512))
window: $(@bind fe_type Select([rect => "rect", hanning => "hanning", hamming => "hamming"]))
"""

# ╔═╡ 9810e7ad-3e0d-46b6-916e-2de73449decc
fe = let fr = Frames(audio; winsize=fe_winsize, winstep=fe_winsize ÷ 2, type=fe_type)
	T = Dict("Dwt" => Dwt, "Wpt" => Wpt, "Swt" => Swt)[fe_kind]
	T(fr; wavelet=fe_wave, level=fe_level, spectrum=fe_spec)
end

# ╔═╡ 984f220c-7d57-4f27-a8c2-d2d73a8952fb
let z = 10 .* log10.(get_spec(fe) .+ 1f-12) .* (get_spectrum(fe) === magnitude ? 2 : 1)
	# plot on band index so the octave bands of Dwt/Swt get equal height
	heatmap(get_times(fe), 1:get_nbands(fe), z; clims=(maximum(z) - 80, maximum(z)),
	        yticks=(1:get_nbands(fe), [string(round(Int, f)) for f in get_freq(fe)]),
	        xguide="Time (s)", yguide="band centre (Hz)", colorbar_title="dB", title=string(fe))
end

# ╔═╡ 5297a341-197e-4217-86d7-c9d51c2155f6
plot(SpectralCentroid(fe); title="SpectralCentroid computed on the $fe_kind front end")

# ╔═╡ 2d76f406-21e6-4ec7-8e12-bcfb36e22021
md"""
## 7. Empirical mode decomposition — `emd`

`emd(x; max_imfs=10, max_sift=50, sd=0.2) -> (imfs, residual)`

EMD makes no assumption about bands.
It repeatedly *sifts* the signal: draw cubic-spline envelopes through the maxima and the minima (mirrored at both ends), subtract their mean, and repeat until what remains oscillates symmetrically around zero — an **intrinsic mode function** (IMF).
That IMF is removed and the process restarts on the residual.
IMFs come out from the fastest oscillation to the slowest, and `sum(imfs, dims=1)' + residual == x` exactly.

| parameter | effect |
|:--|:--|
| `max_imfs` | upper bound on the number of IMFs; the decomposition also stops when the residual has fewer than three extrema. Lowering it lumps all slow content into the residual. |
| `max_sift` | maximum sifting iterations per IMF. More sifts give cleaner, more symmetric IMFs but can over-flatten their amplitude modulation. |
| `sd` | stop sifting when the normalised squared change between two sifts drops below `sd`. Smaller means stricter IMFs and more sifts (typical 0.2–0.3). |

EMD is global and fairly slow, so it runs on an excerpt.
"""

# ╔═╡ ccdeea45-395b-4658-aebc-cef5173eac77
md"""
excerpt start (s): $(@bind emd_start Slider(0:0.05:1.5; default=0.5, show_value=true))
length (samples): $(@bind emd_len Select([1024, 2048, 4096, 8192, 16384]; default=4096))

max\_imfs: $(@bind max_imfs Slider(1:12; default=8, show_value=true))
max\_sift: $(@bind max_sift Slider(1:100; default=50, show_value=true))
sd: $(@bind sd Slider(0.01:0.01:1.0; default=0.2, show_value=true))
"""

# ╔═╡ dfbfaf3c-8bac-43dc-8e26-6605958cb9d3
begin
	i0 = clamp(round(Int, emd_start * sr) + 1, 1, max(1, length(x) - emd_len + 1))
	xe = x[i0:min(end, i0 + emd_len - 1)]
	imfs, residual = emd(xe; max_imfs, max_sift, sd)
	(nimfs = size(imfs, 1), completeness_error = maximum(abs, vec(sum(imfs; dims=1)) .+ residual .- xe))
end

# ╔═╡ 6d5ae9f0-cda9-471f-ab0e-e4c8dbbf07d1
let t = (i0 - 1 .+ (0:length(xe)-1)) ./ sr
	ps = [plot(t, xe; label="", title="signal", titlefontsize=8, c=:black)]
	for k in axes(imfs, 1)
		push!(ps, plot(t, imfs[k, :]; label="", title="IMF $k", titlefontsize=8))
	end
	push!(ps, plot(t, residual; label="", title="residual", titlefontsize=8, c=:gray))
	plot(ps...; layout=(length(ps), 1), size=(720, 80 * length(ps)))
end

# ╔═╡ 26625c62-1740-4d80-b8d7-3add2e5e3e8f
md"""
### Instantaneous frequency

Each IMF is narrow-band, so its analytic signal `hilbert(imf)` has a meaningful phase: its derivative is the instantaneous frequency.
Well-separated IMFs occupy distinct frequency ranges; on speech you see the formant/pitch structure split across IMFs, on the chirp the first IMF follows the sweep.

IMF to play: $(@bind imf_k Slider(1:max(1, size(imfs, 1)); default=1, show_value=true))
"""

# ╔═╡ 2c3d542c-14d6-4c17-b2d9-c22c097833f2
let t = (i0 - 1 .+ (1:length(xe)-1)) ./ sr
	p = plot(xguide="Time (s)", yguide="Hz", title="instantaneous frequency of each IMF (smoothed)", legend=:outerright)
	sm(v, n=31) = [Audio911.mean(@view v[max(1, i - n ÷ 2):min(end, i + n ÷ 2)]) for i in eachindex(v)]
	for k in axes(imfs, 1)
		z = hilbert(imfs[k, :])
		f = abs.(rem2pi.(diff(angle.(z)), RoundNearest)) .* sr ./ 2π
		plot!(p, t, sm(f); label="IMF $k", lw=k == imf_k ? 3 : 1)
	end
	p
end

# ╔═╡ ed923698-1911-49a8-ae54-c5bd73f5aae6
size(imfs, 1) ≥ imf_k ? audio_player(imfs[imf_k, :], sr) : md"no IMF"

# ╔═╡ 2b26e60a-f978-4e5f-81c1-04da9df0e8aa
md"""
## 8. Hilbert-Huang spectrum — `Hht`

`Hht(frames; nbins=winsize ÷ 2 + 1, max_imfs=10, max_sift=50, sd=0.2, spectrum=power)`

The Hilbert-Huang spectrum puts every IMF's instantaneous power `a²` (or amplitude for `spectrum=magnitude`) in the bin of its instantaneous frequency, then averages over each frame with the frame window.
Unlike an STFT its frequency resolution is not limited by the frame length — each sample contributes a single, sharp frequency — but it is only as clean as the IMFs.
`get_imfs(h)` returns the IMFs it used.

| parameter | effect |
|:--|:--|
| `nbins` | number of linear frequency bins from 0 to `sr/2`: purely the display grid |
| `max_imfs`, `max_sift`, `sd` | passed to `emd` (section 7) |
| `spectrum` | `power` or `magnitude` weighting |
| `winsize`, `winstep`, `type` (via `Frames`) | pooling grid only; the frequency axis does not depend on them |
"""

# ╔═╡ 2200cf6a-bde1-4829-8c2c-68ba579e295b
md"""
nbins: $(@bind hht_nbins Select([65, 129, 257, 513]; default=257))
winsize: $(@bind hht_winsize Select([256, 512, 1024]; default=512))
spectrum: $(@bind hht_spec Select([power => "power", magnitude => "magnitude"]))
max\_imfs: $(@bind hht_imfs Slider(1:12; default=8, show_value=true))
"""

# ╔═╡ 3370ef1f-a8a8-4541-a769-603490df2faf
hht = Hht(Frames(xe, sr; winsize=hht_winsize, winstep=hht_winsize ÷ 4, type=hanning);
          nbins=hht_nbins, max_imfs=hht_imfs, max_sift, sd, spectrum=hht_spec)

# ╔═╡ d4fb3111-e70b-46f7-9261-6438dbcf5c02
plot(hht; top_db=60, title="Hilbert-Huang spectrum ($(size(get_imfs(hht), 1)) IMFs), excerpt from $(emd_start) s")

# ╔═╡ aee4cc72-a788-4963-ba35-b4d9e491256e
plot(Stft(Frames(xe, sr; winsize=hht_winsize, winstep=hht_winsize ÷ 4, type=hanning)); top_db=60, title="STFT of the same excerpt, for comparison")

# ╔═╡ 4811362c-d080-43d6-9d9a-e3fe1cc7f41b
md"""
## 9. Empirical wavelet transform — `ewt`, `Ewt`

`ewt(x, sr; nbands=5) -> (components, bounds)`

The EWT (Gilles 2013) builds a wavelet filter bank adapted to the signal: it finds the `nbands` largest local maxima of the Fourier magnitude, places band boundaries halfway between neighbouring peaks, and designs Meyer-type filters on those segments.
The filters form a tight frame, so the components add back to the signal.

| parameter | effect |
|:--|:--|
| `nbands` | number of components (≥ 2). Each extra band splits off the next-strongest spectral peak. It fails if the spectrum has fewer local maxima than `nbands`. |

Because it uses the **raw** spectrum's largest peaks, it shines on signals made of a few strong partials (try *two close tones* or *C major chord*).
On speech the largest raw maxima all sit in the lowest few hertz, so every boundary lands near DC and one component holds almost everything — a property of the method's peak rule, visible in the plot below.
"""

# ╔═╡ 12057e87-c56c-48c4-868c-9f4d9e2873ce
md"""
nbands: $(@bind ewt_n Slider(2:10; default=4, show_value=true))
"""

# ╔═╡ c9dd8f3c-d020-431b-8860-f11daf9941ba
comps, bounds = ewt(x, sr; nbands=ewt_n)

# ╔═╡ 868ccf7a-3dd9-4df2-ab05-e8ab1aaced1d
let X = abs.(Audio911.FFTW.rfft(x)), f = range(0, sr / 2; length=length(x) ÷ 2 + 1)
	p = plot(f, 20 .* log10.(X ./ maximum(X) .+ 1e-8); label="|X|", xguide="Hz", yguide="dB", ylims=(-100, 5),
	         title="spectrum and EWT boundaries", xscale=:log10, xlims=(5, sr / 2))
	vline!(p, bounds; label="boundaries", c=:red, ls=:dash)
	q = plot(title="components", legend=:outerright)
	t = (0:length(x)-1) ./ sr
	for n in axes(comps, 1)
		plot!(q, t, comps[n, :] .+ 2(n - 1); label="band $n")
	end
	plot(p, q; layout=(2, 1), size=(720, 520))
end

# ╔═╡ 0f3df894-d661-426b-adb1-7a5b491915d3
(bounds_Hz = bounds, reconstruction_error = maximum(abs, vec(sum(comps; dims=1)) .- x))

# ╔═╡ 8d81c55a-6fe9-481a-807c-7c45b61551ad
md"""
Component to play: $(@bind ewt_k Slider(1:ewt_n; default=1, show_value=true))
"""

# ╔═╡ b05f864c-352d-4b3e-9456-6acb79671b17
audio_player(comps[min(ewt_k, size(comps, 1)), :], sr)

# ╔═╡ d5d28c07-9279-4a76-b2df-ae35e6e7e414
md"""
`Ewt(frames; nbands=5, spectrum=power)` pools the components on the frames like `Dwt`, returning a `DiscreteWavelet` whose `get_freq` are the centres of the adaptive bands:
"""

# ╔═╡ 672bd043-7b25-4bd4-9352-3912974c8b50
let e = Ewt(Frames(audio; winsize=512, winstep=256); nbands=ewt_n)
	z = 10 .* log10.(get_spec(e) .+ 1f-12)
	heatmap(get_times(e), 1:get_nbands(e), z; clims=(maximum(z) - 80, maximum(z)),
	        yticks=(1:get_nbands(e), [string(round(Int, f)) for f in get_freq(e)]),
	        xguide="Time (s)", yguide="band centre (Hz)", colorbar_title="dB", title=string(e))
end

# ╔═╡ a95237ef-66a5-462e-a046-cf34b2219dff
md"""
## Next

* **7 — Music and rhythm**: chroma, tonnetz, onsets, tempo, beat tracking and harmonic/percussive separation.
* **8 — Time-domain features and pitch**: RMS, zero crossings and the pitch estimators.
* **9 — Signal processing and classic methods**: conversions, time stretch, pitch shift, NMF and HMMs.

Back to **5 — Time-frequency transforms** for the continuous wavelet transform, CQT and friends.
"""

# ╔═╡ Cell order:
# ╟─f3474cc5-d683-4319-92ae-c8686cde6028
# ╟─9a6d4816-d46c-4865-b80b-d85676ffcf0f
# ╟─5596a305-1317-4e36-861b-93d81fc0de9d
# ╟─15d129cf-98ec-4a6e-b2d0-606c55c73e16
# ╠═dce70742-103f-4921-8724-9498c7fa8022
# ╠═fcc6a839-f88b-46ad-9680-f2fdc2197f8a
# ╟─99d66a33-4b6d-4c26-9c95-641a9ad974d4
# ╟─5618323a-be51-434a-ad1a-39d36e4b185d
# ╠═6057b0fb-427b-4cb7-8a6d-8201dda7f9c8
# ╟─79f51240-4732-46b4-baee-93a4172c3d10
# ╟─a833717e-67e3-438a-949d-9c2580a9f08b
# ╠═1779345e-2060-4d5e-82b2-0c36eaff2a94
# ╠═eb52c141-08ed-49c8-8cf5-1a9e1e879443
# ╟─711e7075-2352-44eb-a66d-59fafe070332
# ╠═b07579eb-dbfe-43b3-8b4a-b2177ca4dfa0
# ╠═4097dc44-8203-486c-adc0-291410bfdea3
# ╟─c94f5d5c-6c53-4919-8bd8-06c9765e08ab
# ╠═dde968bb-4d2e-4ea6-a52a-b00f2f1ff30b
# ╟─780f73b7-29b0-4a35-aff4-4fa41a671b11
# ╠═72c19158-939e-4f04-a6c2-016168dd5f9b
# ╠═1493693e-7d3c-4520-9696-319efc54886d
# ╠═aaa6b09a-2baf-42d9-ab44-55679e35b9d9
# ╟─7ac6903e-9999-43d4-a224-b317548f3a20
# ╟─edf3f78e-8472-4726-b5b0-aab74f6c9131
# ╠═78191453-da25-4d5e-90d1-39be7be29690
# ╠═ec4a1820-04dd-4f09-a324-a2d7d330c510
# ╠═dddd2a5a-0bc3-46fb-adf0-9cf98b75fbc5
# ╟─91627684-cffe-4866-8230-1964f5e2f0f3
# ╟─812f1faa-b305-42e6-8985-8b7d2b528a0d
# ╠═a3076844-0223-46c4-ac0d-01870f545f6b
# ╟─05b413f5-6380-45f1-bc6a-00c37e7ccaf7
# ╟─eeeaf1ba-a03c-419d-887e-f36eae9529ef
# ╟─102e0047-cf7e-4578-bc69-a0130b7b54e6
# ╠═7b2440f6-4b80-4a59-be1b-7a01a0f4b915
# ╟─9cff018e-1d32-419e-b13a-1636194f4853
# ╠═26028605-2977-4d61-b31b-b0f8a51af9c2
# ╟─7fe6910d-a9a7-490f-b332-bec537c7a29f
# ╟─7bcac67d-1a62-43ab-bbf1-316b3a36b5d8
# ╠═9810e7ad-3e0d-46b6-916e-2de73449decc
# ╠═984f220c-7d57-4f27-a8c2-d2d73a8952fb
# ╠═5297a341-197e-4217-86d7-c9d51c2155f6
# ╟─2d76f406-21e6-4ec7-8e12-bcfb36e22021
# ╟─ccdeea45-395b-4658-aebc-cef5173eac77
# ╠═dfbfaf3c-8bac-43dc-8e26-6605958cb9d3
# ╠═6d5ae9f0-cda9-471f-ab0e-e4c8dbbf07d1
# ╟─26625c62-1740-4d80-b8d7-3add2e5e3e8f
# ╠═2c3d542c-14d6-4c17-b2d9-c22c097833f2
# ╠═ed923698-1911-49a8-ae54-c5bd73f5aae6
# ╟─2b26e60a-f978-4e5f-81c1-04da9df0e8aa
# ╟─2200cf6a-bde1-4829-8c2c-68ba579e295b
# ╠═3370ef1f-a8a8-4541-a769-603490df2faf
# ╠═d4fb3111-e70b-46f7-9261-6438dbcf5c02
# ╠═aee4cc72-a788-4963-ba35-b4d9e491256e
# ╟─4811362c-d080-43d6-9d9a-e3fe1cc7f41b
# ╟─12057e87-c56c-48c4-868c-9f4d9e2873ce
# ╠═c9dd8f3c-d020-431b-8860-f11daf9941ba
# ╠═868ccf7a-3dd9-4df2-ab05-e8ab1aaced1d
# ╠═0f3df894-d661-426b-adb1-7a5b491915d3
# ╟─8d81c55a-6fe9-481a-807c-7c45b61551ad
# ╠═b05f864c-352d-4b3e-9456-6acb79671b17
# ╟─d5d28c07-9279-4a76-b2df-ae35e6e7e414
# ╠═672bd043-7b25-4bd4-9352-3912974c8b50
# ╟─a95237ef-66a5-462e-a046-cf34b2219dff
