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

# ╔═╡ 9caa31f3-5baa-4845-9db2-739bf7135b72
begin
	import Pkg
	# the notebook environment next to this file pins Audio911 (from this repository), PlutoUI and Plots
	Pkg.activate(@__DIR__)
	Pkg.instantiate()
	using Audio911, PlutoUI, Plots
	include(joinpath(@__DIR__, "common.jl"))
	gr(); default(size=(720, 300), legend=:topright, titlefontsize=10, guidefontsize=9)
end

# ╔═╡ ffb26580-b763-4ce9-bf32-dea9a0e5f870
md"""
# Audio911 · 5 — Time-frequency front ends beyond the STFT

The STFT (notebook 2) analyses every frequency with the same window, so its time and frequency resolution are fixed everywhere.
The front ends in this notebook relax that constraint in different ways:

| front end | idea | resolution |
|:--|:--|:--|
| `Cwt` | continuous wavelet transform: one analytic wavelet stretched to every scale | long analysis at low frequencies, short at high |
| `Cqt` | constant-Q transform: a windowed complex exponential per bin, `Q` cycles long | same trade-off as `Cwt`, on a musical grid |
| `Pwt` | pseudo wavelet transform: an auditory filterbank applied to the spectrum of the whole signal | set by the filterbank bands |
| `St` / `Fst` | Stockwell transform: Gaussian windows whose width scales with frequency | frequency-dependent, phase referenced to time zero |
| `Nsgt` | non-stationary Gabor transform: one invertible Gabor frame per band | every band sampled at its own rate |
| `Synsq` / `Wsst` | synchrosqueezing: move every wavelet coefficient to its instantaneous frequency | much sharper ridges than the `Cwt` |
| `Reassign` | reassigned spectrogram: move every STFT cell to its centre of gravity | much sharper than the `Stft` |
| `Wvd` / `Cwd` | Cohen-class quadratic distributions (Wigner-Ville, Choi-Williams) | twice the STFT's resolution, with interference terms |

Every one of them is an `AbstractSpectrogram` that is **pooled onto the time grid of a `Frames` object**: one column per frame, exactly like `Stft(frames)`.
That is why `MelSpec`, `Mfcc`, the spectral descriptors and `Chroma` accept any of them unchanged (last section).
"""

# ╔═╡ 52ed1fb7-77f8-4cb4-836f-631378043e6e
TableOfContents()

# ╔═╡ e0810837-579a-45f9-997d-260262790866
md"""
## 1. Signal and time grid

All front ends below share one time grid: the frames built here.
The *two close tones and clicks* signal is the best test of resolution (can the two tones at 440 and 470 Hz be separated, and are the clicks sharp?), the *chirp* shows how each transform follows a moving frequency.

| parameter | effect |
|:--|:--|
| `winsize` | for the pooled transforms (`Cwt`, `Pwt`, `St`, `Nsgt`, `Wsst`) it only sets the pooling length, **not** the analysis resolution; for `Reassign`, `Wvd` and `Cwd` it is the analysis window itself. |
| `winstep` | the hop, i.e. the number of columns per second of every front end. |

The wavelet-type transforms pool with a rectangular window (the `Cwt(audio)` default), the Fourier-type ones use a Hann window.
"""

# ╔═╡ 76a120a4-3dec-40f6-8d01-36207edf35db
md"""
Signal: $(@bind signal_name Select(SIGNALS; default="tones"))
Sample rate: $(@bind sr Select([16000 => "16 kHz", 8000 => "8 kHz"]))
winsize: $(@bind winsize Select([128, 256, 512, 1024]; default=512))
hop: $(@bind winstep Select([32, 64, 128, 256]; default=128))
"""

# ╔═╡ 80f572a1-e5a7-4799-b952-25bdcb75ced2
audio = demo_signal(signal_name; sr)

# ╔═╡ cf94450c-2e04-4de2-bf67-9e5dff9ce04b
audio_player(audio)

# ╔═╡ 42e8e4fc-790f-49ae-97b8-5936670545b3
begin
	frames_rect = Frames(audio; winsize, winstep, type=rect)
	frames_hann = Frames(audio; winsize, winstep, type=hanning)
	(frames = length(frames_rect), columns_per_second = sr / winstep)
end

# ╔═╡ 71baf269-918c-44cc-97e9-3275acf323dd
md"""
## 2. Continuous wavelet transform — `Cwt`

`Cwt(frames; kwargs...)` multiplies the spectrum of the whole signal by the analytic wavelet `ψ̂(s·ω)` at every scale `s`, inverse transforms it and pools `|W|²` over the frames.
Low frequencies get long wavelets (sharp in frequency, blurred in time), high frequencies short ones.

| parameter | effect |
|:--|:--|
| `wavelet` | mother wavelet (Fourier transform `ω -> ψ̂(ω)`). Its bandwidth sets the time/frequency trade-off at every scale; see the shapes below. Any function of the angular frequency works. |
| `voices` | scales per octave on the default geometric grid. More voices = finer frequency sampling (more rows), not better resolution. |
| `freqrange` | lowest and highest centre frequency (Hz). The lowest scale dominates the cost: its wavelet is the longest. |
| `scale`, `nbands`, `bins_per_octave` | instead of `voices`, place the bands on a filterbank grid (`octave`, `logspace`, `linspace`, `htk`, `erb`, …), as audioFlux does. |
| `centre` | angular frequency of the wavelet at unit scale, which maps scale to Hz. Defaults to audioFlux's value for the built-in wavelets and to the spectral peak for other functions. |
| `spectrum` | `power` (`|W|²`) or `magnitude` (`|W|`). |

Wavelet shape parameters (passed by wrapping the wavelet, e.g. `ω -> morlet(ω; ω0=8)`):

| wavelet | parameters | effect |
|:--|:--|:--|
| `morlet` | `ω0=6` | centre frequency in cycles; larger = more oscillations = sharper in frequency, blurrier in time |
| `morse` | `β=20`, `γ=3` | `γ` the symmetry (3 ≈ Gaussian-like), `β·γ` the time-bandwidth product (larger = narrower in frequency) |
| `bump` | `μ=5`, `σ=0.6` | compact support in frequency, `σ` the bandwidth: the sharpest frequency localisation |
| `paul` | `m=4` | order; low orders are very short in time (good for transients) |
| `dog` | `m=2`, `β=2` | derivative-of-Gaussian of even order `m`; real, broadband |
| `mexican` | `β=2` | `dog` of order 2, the Mexican hat |
| `hermit` | `γ=5`, `β=2` | Hermitian wavelet centred at `γ`, `β` the width |
| `ricker` | `γ=4` | peaks at `ω = γ` |

Changing a parameter from its default passes a closure, so the scale-to-Hz mapping falls back to the spectral peak of the modified wavelet.
"""

# ╔═╡ 7199a519-63d2-49e3-a326-9e4e38becd66
md"""
wavelet: $(@bind wname Select(["morlet", "morse", "bump", "paul", "dog", "mexican", "hermit", "ricker"]))
grid: $(@bind cwt_grid Select(["voices" => "geometric (voices)", "octave" => "octave", "logspace" => "logspace", "linspace" => "linspace", "htk" => "mel (htk)", "erb" => "erb"]))
spectrum: $(@bind cwt_spectrum Select([power => "power", magnitude => "magnitude"]))

voices: $(@bind voices Slider(2:2:32; default=12, show_value=true))
nbands (filterbank grids): $(@bind cwt_nbands Slider(16:4:128; default=64, show_value=true))
bins per octave (octave grid): $(@bind cwt_bpo Slider(6:6:48; default=12, show_value=true))

fmin (Hz): $(@bind cwt_fmin Slider([33, 50, 100, 200, 400]; default=50, show_value=true))
fmax (Hz): $(@bind cwt_fmax Slider([1000, 2000, 4000, 6000, 8000]; default=4000, show_value=true))
"""

# ╔═╡ eb0a1f65-d7c5-4fd9-86f4-35c5d0af351f
function wavelet_ui(name)
	s(r, d) = Slider(r; default=d, show_value=true)
	PlutoUI.combine() do Child
		name == "morlet"  ? md"ω0 $(Child(:ω0, s(3:0.5:14, 6)))" :
		name == "morse"   ? md"β $(Child(:β, s(2:2:60, 20))) γ $(Child(:γ, s(1:0.5:6, 3)))" :
		name == "bump"    ? md"μ $(Child(:μ, s(2:0.5:10, 5))) σ $(Child(:σ, s(0.1:0.1:1.5, 0.6)))" :
		name == "paul"    ? md"m $(Child(:m, s(1:1:16, 4)))" :
		name == "dog"     ? md"m $(Child(:m, s(2:2:12, 2))) β $(Child(:β, s(0.5:0.5:6, 2)))" :
		name == "mexican" ? md"β $(Child(:β, s(0.5:0.5:6, 2)))" :
		name == "hermit"  ? md"γ $(Child(:γ, s(2:0.5:10, 5))) β $(Child(:β, s(0.5:0.5:6, 2)))" :
		                    md"γ $(Child(:γ, s(1:0.5:10, 4)))"
	end
end

# ╔═╡ ef1f470e-26d7-4b61-856c-3e9e5cc3cbdb
@bind wpar wavelet_ui(wname)

# ╔═╡ a0ee1eb2-a0ff-43ce-89a5-ec0598024a00
# the wavelet as a function of ω: the named function at its defaults, a closure otherwise
function make_wavelet(name, p)
	defaults = Dict("morlet" => (ω0=6,), "morse" => (β=20, γ=3), "bump" => (μ=5, σ=0.6),
		"paul" => (m=4,), "dog" => (m=2, β=2), "mexican" => (β=2,),
		"hermit" => (γ=5, β=2), "ricker" => (γ=4,))
	f = getfield(Audio911, Symbol(name))
	all(Float64(getfield(p, k)) == Float64(v) for (k, v) in pairs(defaults[name])) && return f
	kw = NamedTuple{keys(p)}(Tuple(k === :m ? Int(v) : v for (k, v) in pairs(p)))
	return ω -> f(ω; kw...)
end

# ╔═╡ 4df98457-9f16-4aad-8dbb-08b1e08cbb57
wavelet = make_wavelet(wname, wpar)

# ╔═╡ 40bb15b8-f619-4574-9373-2d00fc227bc7
cwt_fr = let
	fmax = min(cwt_fmax, sr ÷ 2)
	if cwt_grid == "voices"
		Cwt(frames_rect; wavelet, voices, freqrange=(cwt_fmin, fmax), spectrum=cwt_spectrum)
	else
		Cwt(frames_rect; wavelet, scale=getfield(Audio911, Symbol(cwt_grid)), nbands=cwt_nbands,
		    bins_per_octave=cwt_bpo, freqrange=(cwt_fmin, fmax), spectrum=cwt_spectrum)
	end
end

# ╔═╡ 324f4360-63e5-4064-9395-69d8d99f6122
plot(cwt_fr; freq_scale=:log10, title="Cwt, $wname, $(get_nbins(cwt_fr)) bands")

# ╔═╡ 255efd28-9a3e-4987-b396-861fe7b8e5bf
md"""
### The eight mother wavelets

Left: the Fourier transform `ψ̂(ω)` (all are analytic: zero for negative frequencies). Right: the real part of the wavelet in time, at a scale where it peaks at 1/8 of the sample rate.
A wide `ψ̂` is a short wavelet; `bump` is the narrowest in frequency, `paul` and `dog` the shortest in time.
The wavelet selected above is drawn thick.
"""

# ╔═╡ c805020f-3077-4c90-b75d-cbf598502f05
let ω = range(0, 20; length=800), n = 256
	p1 = plot(title="ψ̂(ω)", xguide="ω (rad/sample at unit scale)")
	p2 = plot(title="Re ψ(t)", xguide="sample")
	for name in ["morlet", "morse", "bump", "paul", "dog", "mexican", "hermit", "ricker"]
		f = name == wname ? wavelet : getfield(Audio911, Symbol(name))
		v = f.(ω); v ./= maximum(abs, v)
		lw = name == wname ? 3 : 1
		plot!(p1, ω, v; label=name, lw)
		# evaluate on the FFT grid, scaled so the spectral peak sits at π/4 rad/sample
		ωpk = ω[argmax(abs.(v))]
		k = 0:n-1
		Ψ = ComplexF64[kk ≤ n ÷ 2 ? f(2π * kk / n * ωpk / (π / 4)) : 0 for kk in k]
		ψ = Audio911.FFTW.ifft(Ψ)
		ψ = circshift(ψ, n ÷ 2)
		plot!(p2, real.(ψ) ./ maximum(abs, ψ); label=name, lw)
	end
	plot(p1, p2; layout=(1, 2), size=(720, 320), legend=:topright, legendfontsize=6)
end

# ╔═╡ 66473808-11bb-4688-9a78-72d6ea044433
md"""
### Whole-signal `cwt` and the complex coefficients

`cwt(x, sr; kwargs...)` returns the complex transform at **every sample** (`nbands × length(x)`) together with the band frequencies; it uses audioFlux's octave grid from C1 by default.
Extra keywords: `pad=true` pads symmetrically by half the signal length (otherwise the signal is treated as periodic and the edges wrap), `T` the element type.
`get_complex(c::Cwt)` gives the complex coefficients at the frame centres of a pooled `Cwt`.
Below, an excerpt at full time resolution: the modulus, and the real part of the band nearest to the chosen frequency.
"""

# ╔═╡ 180cfe81-1d21-4520-b087-23e03609ea39
md"""
excerpt start (s): $(@bind cwt_t0 Slider(0:0.05:1.5; default=0.2, show_value=true))
band frequency (Hz): $(@bind cwt_band Slider([100, 220, 440, 880, 1760, 3000]; default=440, show_value=true))
pad $(@bind cwt_pad CheckBox(default=true))
"""

# ╔═╡ aeff41ac-3223-4958-8e0a-32c7e0cc4b81
let
	x = vec(get_data(audio))
	i0 = round(Int, cwt_t0 * sr) + 1
	seg = x[i0:min(end, i0 + sr ÷ 4)]
	W, freq = cwt(seg, sr; wavelet, bins_per_octave=24, nbands=floor(Int, 24 * log2((sr / 2) / 33)) - 1, freqrange=(33, sr ÷ 2), pad=cwt_pad)
	t = cwt_t0 .+ (0:length(seg)-1) ./ sr
	A = 20 .* log10.(max.(abs.(W) ./ maximum(abs, W), 1e-4))
	p1 = heatmap(t, freq, A; yscale=:log10, xguide="Time (s)", yguide="Hz", title="|cwt| (dB), every sample", colorbar_title="dB")
	k = argmin(abs.(freq .- cwt_band))
	p2 = plot(t, real.(W[k, :]); label="Re W at $(round(freq[k]; digits=1)) Hz", xguide="Time (s)")
	plot!(p2, t, abs.(W[k, :]); label="|W|", ls=:dash)
	plot(p1, p2; layout=(2, 1), size=(720, 480))
end

# ╔═╡ 3c81ccd1-7771-400b-bdce-f92d84f58577
(complex_at_frame_centres = size(get_complex(cwt_fr)), eltype = eltype(get_complex(cwt_fr)))

# ╔═╡ cf77b2a8-f7b9-4e78-9a7e-1cd424196709
md"""
## 3. Constant-Q transform — `Cqt`

`Cqt(frames; kwargs...)` uses, for every bin `k`, a windowed complex exponential of `Q · sr / f_k` samples, so every bin spans the same number of cycles.
The frames only fix the time grid: the analysis windows are centred on the frame centres and can be much longer than `winsize` at low frequencies.

| parameter | effect |
|:--|:--|
| `fmin` | frequency of the first bin (default C1 = 32.7 Hz). |
| `bins_per_octave` | 12 = semitones, 24 = quarter tones, 36 = the usual choice for pitch tracking. Also sets `Q`: more bins per octave = longer kernels = sharper frequency, blurrier time. |
| `nbins` | number of bins; `nbins / bins_per_octave` octaves above `fmin`. The top bin must stay below `sr/2`. |
| `factor` | scales `Q`. `< 1` shortens every kernel (better time resolution, broader bins). |
| `gamma` | variable-Q offset in Hz (Schörkhuber et al.). `0` is constant-Q; `> 0` shortens the low-frequency kernels, giving the low bins better time resolution. |
| `thresh` | sparsity threshold of the spectral kernels; larger is faster and less accurate. |
| `window` | kernel window (`hanning`, `hamming`, `blackman`, …): main-lobe width vs leakage, as in notebook 1. |
| `norm` | kernel normalisation: `area` (unit L1), `none_norm` (divide by the length) or `bandwidth`. Changes the relative level of low and high bins. |
| `scale` | divide every bin by the square root of its kernel length. |
| `spectrum` | `power` or `magnitude`. |
| `keep_complex` | keep the complex coefficients for `get_complex` instead of recomputing them. |
"""

# ╔═╡ 11ee93c2-de75-40b4-ab20-fa539b2130da
md"""
fmin: $(@bind cqt_fmin Select([32.703 => "C1 (32.7 Hz)", 65.406 => "C2 (65.4 Hz)", 130.81 => "C3 (130.8 Hz)"]))
bins per octave: $(@bind cqt_bpo Select([12, 24, 36, 48]))
octaves: $(@bind cqt_oct Slider(3:7; default=6, show_value=true))

factor: $(@bind cqt_factor Slider(0.25:0.25:2; default=1, show_value=true))
gamma (Hz): $(@bind cqt_gamma Slider(0:5:100; default=0, show_value=true))
thresh: $(@bind cqt_thresh Select([0.0, 0.001, 0.01, 0.05]; default=0.01))

window: $(@bind cqt_window Select([hanning => "hanning", hamming => "hamming", blackman => "blackman", rect => "rect"]))
norm: $(@bind cqt_norm Select([area => "area", none_norm => "none_norm", bandwidth => "bandwidth"]))
scale $(@bind cqt_scale CheckBox(default=true))
spectrum: $(@bind cqt_spectrum Select([power => "power", magnitude => "magnitude"]))
"""

# ╔═╡ a239f653-4508-419d-9d9a-eeaa44d18be8
cqt = let
	nbins = cqt_bpo * cqt_oct
	# keep the top bin below Nyquist
	while cqt_fmin * 2^((nbins - 1) / cqt_bpo) ≥ sr / 2
		nbins -= cqt_bpo
	end
	Cqt(frames_hann; fmin=cqt_fmin, bins_per_octave=cqt_bpo, nbins, factor=cqt_factor,
	    gamma=cqt_gamma, thresh=cqt_thresh, window=cqt_window, norm=cqt_norm,
	    scale=cqt_scale, spectrum=cqt_spectrum, keep_complex=true)
end

# ╔═╡ 8227debd-b7c6-4974-bc27-024520c7194e
plot(cqt)

# ╔═╡ de98263f-667b-4c9e-abf6-1ea97667324c
let
	L = get_bandwidth(cqt)
	plot(get_freq(cqt), 1000 .* L ./ sr; xscale=:log10, yscale=:log10, xguide="Hz", yguide="kernel length (ms)",
	     label="factor=$cqt_factor, gamma=$cqt_gamma", title="Cqt kernel length per bin (nfft of the lowest octave = $(get_nfft(cqt)))")
	plot!(get_freq(cqt), fill(1000 * winsize / sr, length(L)); label="winsize of the frames", ls=:dash)
end

# ╔═╡ 5993953d-6eda-43de-b0db-93bff734f8c3
md"""
The kernel length plot shows the whole trade-off: with `gamma = 0` it halves every octave (constant Q); raising `gamma` flattens the low end.
Kernels longer than the dashed line see well beyond one frame, which is why the low bins smear the clicks.
"""

# ╔═╡ fa8eced5-3a8e-4412-a9de-ac3233485597
md"""
## 4. Pseudo wavelet transform — `Pwt`

`Pwt(frames; kwargs...)` takes the FFT of the **whole** signal, multiplies it by every band of an auditory filterbank designed on that fine frequency grid, and inverse transforms each band (audioFlux `PWT`).
Each band is an analytic signal whose time resolution is the inverse of the band's width.
`pwt(x, sr; kwargs...)` returns the complex band series at every sample.

| parameter | effect |
|:--|:--|
| `scale` | band spacing: `octave` (musical, default), `logspace`, `linspace`, `htk`/`slaney` (mel), `bark`, `erb`. |
| `nbands` | number of bands. On a fixed range, more bands = narrower bands = sharper frequency, blurrier time. |
| `bins_per_octave` | bands per octave for the `octave` scale. |
| `freqrange` | range covered by the bands. |
| `style` | band shape: `triangular`, `hanning`, `rect`, `gauss`, `kaiser`, `bohman`, `point`, … Smooth shapes ring less in time. |
| `norm` | band normalisation: `none_norm` (peak 1), `area`, `bandwidth`. Changes the relative level of wide and narrow bands. |
| `spectrum` | `power` or `magnitude` of the pooled result. |
"""

# ╔═╡ 9ff7950a-9a15-4eeb-ae18-b32fa9278ad2
md"""
scale: $(@bind pwt_scale Select([octave => "octave", logspace => "logspace", linspace => "linspace", htk => "mel (htk)", slaney => "mel (slaney)", bark => "bark", erb => "erb"]))
style: $(@bind pwt_style Select([triangular => "triangular", hanning => "hanning", rect => "rect", gauss => "gauss", kaiser => "kaiser", bohman => "bohman", blackman => "blackman", point => "point"]))
norm: $(@bind pwt_norm Select([none_norm => "none_norm", area => "area", bandwidth => "bandwidth"]))

nbands: $(@bind pwt_nbands Slider(12:6:120; default=72, show_value=true))
bins per octave (octave): $(@bind pwt_bpo Slider(6:6:24; default=12, show_value=true))
spectrum: $(@bind pwt_spectrum Select([power => "power", magnitude => "magnitude"]))
"""

# ╔═╡ 6eb2d809-3e10-4e14-ae28-38cd53ca4567
pwt_fr = let
	fr = pwt_scale in (octave, logspace) ? (33, sr ÷ 2) : (0, sr ÷ 2)
	nb = pwt_scale === octave ? min(pwt_nbands, floor(Int, log2((sr / 2) / 32.703) * pwt_bpo) - 1) : pwt_nbands
	Pwt(frames_rect; nbands=nb, scale=pwt_scale, style=pwt_style, norm=pwt_norm,
	    freqrange=fr, bins_per_octave=pwt_bpo, spectrum=pwt_spectrum)
end

# ╔═╡ b60a913a-bf9c-42d8-91dd-b779291a4efa
plot(pwt_fr; freq_scale=(pwt_scale in (octave, logspace) ? :log10 : :linear))

# ╔═╡ 15e52fa2-00de-43be-885a-cc8179dfa3b8
let x = vec(get_data(audio))[1:min(end, sr ÷ 2)]
	Y, freq = pwt(x, sr; nbands=get_nbins(pwt_fr), scale=pwt_scale, style=pwt_style, norm=pwt_norm,
	              freqrange=get_setup(pwt_fr).freqrange, bins_per_octave=pwt_bpo)
	heatmap((0:length(x)-1) ./ sr, 1:length(freq), 20 .* log10.(max.(abs.(Y) ./ maximum(abs, Y), 1e-4));
	        xguide="Time (s)", yguide="band index", title="pwt(x, sr): |Y| (dB) at every sample, first 0.5 s")
end

# ╔═╡ fcac67c2-eff6-4d99-883d-2ec50ae0d005
md"""
## 5. Stockwell transform — `St` and `Fst`

The S-transform is an STFT whose Gaussian window has a width inversely proportional to frequency, with the phase referenced to time zero.
`st(x; min_index, max_index, factor, norm)` works on the whole signal and costs one inverse FFT of the signal length **per frequency bin**, so it is quadratic in the signal length: this section uses a short excerpt.
`fst(x; min_index, max_index)` is the fast discrete S-transform (Brown et al. 2010) on dyadic blocks; it needs a power-of-two length (`Fst` pads).

| parameter | effect |
|:--|:--|
| `freqrange` (`St`, `Fst`) / `min_index`, `max_index` (`st`, `fst`) | which FFT bins of the whole signal are computed; the grid spacing is `sr / length(x)`, so a longer excerpt gives a finer grid. |
| `factor` | `λ`, scales the Gaussian width. Larger = longer windows = sharper frequency, blurrier time. |
| `norm` | `p`, the exponent of the frequency dependence of the width (`1` = the classic S-transform; smaller makes the windows depend less on frequency, closer to an STFT). |
| `spectrum` | `power` or `magnitude`. |

`Fst` has no shape parameters: its dyadic partition is fixed, which shows as blocky, octave-wide cells.
"""

# ╔═╡ 8b6f570d-ccde-4995-86ec-a433e0362c31
md"""
excerpt length (s): $(@bind st_len Slider(0.1:0.05:0.6; default=0.3, show_value=true))
excerpt start (s): $(@bind st_t0 Slider(0:0.05:1.4; default=0.1, show_value=true))
max frequency (Hz): $(@bind st_fmax Slider([1000, 2000, 4000, 8000]; default=2000, show_value=true))

factor λ: $(@bind st_factor Slider(0.25:0.25:4; default=1, show_value=true))
norm p: $(@bind st_norm Slider(0.25:0.25:1.5; default=1, show_value=true))
"""

# ╔═╡ 9f87bb80-41fd-46b6-af7a-9925d48b6ff9
st_frames = let x = vec(get_data(audio))
	i0 = round(Int, st_t0 * sr) + 1
	seg = x[i0:min(end, i0 + round(Int, st_len * sr) - 1)]
	Frames(seg, sr; winsize=min(winsize, 256), winstep=min(winstep, 64), type=rect)
end

# ╔═╡ efce1f00-7cc0-4189-a1b6-2fb2e42f6329
let fmax = min(st_fmax, sr ÷ 2)
	s = St(st_frames; freqrange=(0, fmax), factor=st_factor, norm=st_norm)
	f = Fst(st_frames; freqrange=(0, fmax))
	plot(plot(s; title="St (λ=$st_factor, p=$st_norm): $(get_nbins(s)) bins"),
	     plot(f; title="Fst: $(get_nbins(f)) bins"); layout=(1, 2), size=(720, 320))
end

# ╔═╡ 795ca5e9-7fb9-49d3-abe5-59c057d0dd47
let x = get_signal(st_frames)[1:min(end, 1024)]
	S = st(x; max_index=length(x) ÷ 4, factor=st_factor, norm=st_norm)
	F = fst(x; max_index=length(x) ÷ 4)
	db(M) = 20 .* log10.(max.(abs.(M) ./ maximum(abs, M), 1e-4))
	t = (0:length(x)-1) ./ sr
	f = (0:size(S, 1)-1) .* (sr / length(x))
	plot(heatmap(t, f, db(S); title="st(x): every sample", xguide="s", yguide="Hz"),
	     heatmap(t, f[1:size(F, 1)], db(F); title="fst(x)", xguide="s"); layout=(1, 2), size=(720, 300))
end

# ╔═╡ b41de932-a681-478a-888a-9973deff2bf1
md"""
## 6. Non-stationary Gabor transform — `Nsgt`

The NSGT (audioFlux `NSGT`) is an invertible transform in which every frequency band has its own window and **its own sampling rate**: narrow low bands get few, long samples, wide high bands many short ones.
`nsgt(x, sr; kwargs...)` returns the raw `cells` (one complex vector per band), the band centres and the cell lengths; `Nsgt(frames)` pools them onto the frames; `get_cells` and `get_lengths` return the raw cells of a pooled `Nsgt`; `nsgt_matrix(cells, lengths, duration, ncols)` is audioFlux's rectangular view of the cells.

| parameter | effect |
|:--|:--|
| `scale`, `nbands`, `bins_per_octave`, `freqrange` | band edges, as for `Pwt`. |
| `style` | window shape of every band in frequency (`hanning` default). |
| `norm` | `bandwidth` (default) divides every window by the square root of its length (tight frame), `area`, `none_norm`. |
| `min_len` | minimum window length in bins: stops the lowest bands from collapsing to a single bin. |
| `standard` | `true` uses the band edges directly (`right − left + 1`, periodic window); `false` a symmetric window around the centre. |
| `spectrum` | `power` or `magnitude` of the pooled result. |
"""

# ╔═╡ 7087450f-db13-4530-8bbf-fdb39bbe60fa
md"""
scale: $(@bind nsgt_scale Select([octave => "octave", logspace => "logspace", linspace => "linspace", htk => "mel (htk)", erb => "erb"]))
style: $(@bind nsgt_style Select([hanning => "hanning", triangular => "triangular", rect => "rect", gauss => "gauss", blackman => "blackman"]))
norm: $(@bind nsgt_norm Select([bandwidth => "bandwidth", area => "area", none_norm => "none_norm"]))

nbands: $(@bind nsgt_nbands Slider(12:6:96; default=72, show_value=true))
bins per octave (octave): $(@bind nsgt_bpo Slider(6:6:24; default=12, show_value=true))
min length: $(@bind nsgt_minlen Slider(1:1:32; default=3, show_value=true))
standard $(@bind nsgt_standard CheckBox(default=false))
"""

# ╔═╡ 0232aa76-ccf0-4bf6-9793-9e2f1dd1cf55
nsgt_fr = let
	fr = nsgt_scale in (octave, logspace) ? (33, sr ÷ 2) : (0, sr ÷ 2)
	nb = nsgt_scale === octave ? min(nsgt_nbands, floor(Int, log2((sr / 2) / 32.703) * nsgt_bpo) - 1) : nsgt_nbands
	Nsgt(frames_rect; nbands=nb, scale=nsgt_scale, style=nsgt_style, norm=nsgt_norm,
	     freqrange=fr, bins_per_octave=nsgt_bpo, min_len=nsgt_minlen, standard=nsgt_standard)
end

# ╔═╡ ebbe242c-2140-40a0-aacc-cf2d6da5a873
let
	lens = get_lengths(nsgt_fr)
	p1 = plot(nsgt_fr; freq_scale=(nsgt_scale in (octave, logspace) ? :log10 : :linear), title="Nsgt pooled on the frames")
	p2 = plot(get_freq(nsgt_fr), lens; xscale=:log10, yscale=:log10, xguide="band centre (Hz)", yguide="samples in the cell",
	          label="", title="samples per band", marker=:circle, ms=2)
	plot(p1, p2; layout=(1, 2), size=(720, 300))
end

# ╔═╡ a573bb5d-2067-4c6c-ae5d-628e3257fb1a
@bind nsgt_band Slider(1:get_nbins(nsgt_fr); default=get_nbins(nsgt_fr) ÷ 2, show_value=true)

# ╔═╡ 42b1035e-084b-491f-baae-85c0702bc892
let
	cells, lens = get_cells(nsgt_fr), get_lengths(nsgt_fr)
	dur = length(get_signal(frames_rect)) / sr
	c = cells[nsgt_band]
	p1 = plot(range(0, dur; length=length(c)), abs.(c); marker=:circle, ms=2, xguide="Time (s)",
	          label="", title="cell $nsgt_band at $(round(get_freq(nsgt_fr)[nsgt_band]; digits=1)) Hz: $(length(c)) samples at its own rate")
	M = nsgt_matrix(cells, lens, dur, maximum(lens))
	p2 = heatmap(range(0, dur; length=size(M, 2)), 1:size(M, 1), 20 .* log10.(max.(abs.(M) ./ maximum(abs, M), 1e-4));
	             xguide="Time (s)", yguide="band", title="nsgt_matrix (hold rule, $(size(M, 2)) columns)")
	plot(p1, p2; layout=(2, 1), size=(720, 480))
end

# ╔═╡ 28f6a6b6-9b40-497c-b287-bc53d4d750d5
md"""
## 7. Synchrosqueezing — `Synsq` and `Wsst`

A wavelet transform smears a pure tone over several neighbouring scales.
Synchrosqueezing estimates the instantaneous frequency of every coefficient and moves the coefficient to the band nearest to that frequency, which concentrates the energy on the ridge.

* `Synsq(cwt)` estimates the frequency from the phase difference between consecutive samples (audioFlux `synsq`).
* `Wsst(cwt)` uses the derivative wavelet, `|Im(W_∂/W)|·sr/2π` (Daubechies, Lu & Wu 2011). `Wsst(frames; kwargs...)` builds the `Cwt` too (Morse wavelet, octave grid by default).
* The whole-signal forms are `synsq(W, freq, sr; scale, thresh, order)` on a complex `cwt` and `wsst(x, sr; ...)`, which returns `(S, W, freq)`.

| parameter | effect |
|:--|:--|
| `thresh` | coefficients with `|W| ≤ thresh` are dropped. Raise it to remove the noise floor, lower it to keep quiet components. |
| `order` | `> 1` repeats the band mapping, squeezing further. |
| `accumulate` | `:energy` adds the power of the moved coefficients (low memory); `:complex` adds the complex values first, as audioFlux does, so coefficients with opposite phase cancel. |
| `spectrum` | `power` or `magnitude`. |

Both use the `Cwt` built in section 2, so try them with different wavelets.
"""

# ╔═╡ 446c3ec8-8bdd-4bc1-a2dc-5eda0ee16495
md"""
thresh: $(@bind sq_thresh Select([0.0, 1e-4, 1e-3, 1e-2, 0.05]; default=1e-3))
order: $(@bind sq_order Slider(1:3; default=1, show_value=true))
accumulate: $(@bind sq_acc Select([:energy, :complex]))
"""

# ╔═╡ 50e1a6ac-9ddd-492a-9b16-55ba82617275
begin
	synsq_fr = Synsq(cwt_fr; thresh=sq_thresh, order=sq_order, accumulate=sq_acc)
	wsst_fr  = Wsst(cwt_fr; thresh=sq_thresh, order=sq_order, accumulate=sq_acc)
	plot(plot(cwt_fr; freq_scale=:log10, title="Cwt ($wname)", colorbar=false),
	     plot(synsq_fr; freq_scale=:log10, title="Synsq", colorbar=false),
	     plot(wsst_fr; freq_scale=:log10, title="Wsst", colorbar=false); layout=(1, 3), size=(760, 300))
end

# ╔═╡ a829525c-72ee-46a6-b9a6-2e7582c99ba0
let x = vec(get_data(audio))[1:min(end, sr ÷ 4)]
	S, W, freq = wsst(x, sr; thresh=sq_thresh, order=sq_order, freqrange=(33, sr ÷ 2))
	Q = synsq(W, freq, sr; thresh=sq_thresh, order=sq_order)
	db(M) = 20 .* log10.(max.(abs.(M) ./ maximum(abs, M), 1e-4))
	t = (0:length(x)-1) ./ sr
	plot(heatmap(t, freq, db(W); yscale=:log10, title="cwt (Morse)", colorbar=false),
	     heatmap(t, freq, db(Q); yscale=:log10, title="synsq(W)", colorbar=false),
	     heatmap(t, freq, db(S); yscale=:log10, title="wsst(x)", colorbar=false); layout=(1, 3), size=(760, 280), xguide="s")
end

# ╔═╡ 899df21d-ad53-45e4-bd53-9f0bc2f0b906
md"""
## 8. Reassigned spectrogram — `Reassign`

`Reassign(stft)` computes, for every STFT cell, the instantaneous frequency (from an STFT with the derivative window) and the group delay (from an STFT with a time-weighted window), and moves the cell's energy there (Auger & Flandrin 1995).
Pure tones collapse onto lines and clicks onto vertical strokes, far below the STFT's resolution limit.
It needs a Fourier front end with a smooth window, so it works on `Stft(frames_hann)`; `get_reassigned(r)` returns the reassigned frequency and time of every cell.

| parameter | effect |
|:--|:--|
| `mode` | `:all` moves cells in time and frequency, `:freq` only in frequency (sharp tones, blurred clicks), `:time` only in time. |
| `thresh` | cells quieter than this stay in place. |
| `order` | `> 1` repeats the frequency reassignment. |
| `accumulate` | `:complex` (audioFlux's default) adds the complex values re-referenced to the window centre; `:energy` adds their power and preserves the total energy. |
| `spectrum` | `power` or `magnitude` of the result. |
| `winsize` (of the frames above) | the STFT window: reassignment sharpens it, but the window still decides which components are resolved at all. |
"""

# ╔═╡ aa134e03-bdd7-4259-91d1-ef3c55378f51
md"""
mode: $(@bind ra_mode Select([:all, :freq, :time]))
thresh: $(@bind ra_thresh Select([0.0, 1e-4, 1e-3, 1e-2]; default=1e-3))
order: $(@bind ra_order Slider(1:3; default=1, show_value=true))
accumulate: $(@bind ra_acc Select([:complex, :energy]))
max frequency shown (Hz): $(@bind ra_fmax Slider([1000, 2000, 4000, 8000]; default=2000, show_value=true))
"""

# ╔═╡ 07531127-5c0f-428b-9f9c-4444cbb88444
begin
	stft_c = Stft(frames_hann; keep_complex=true)
	reassigned = Reassign(stft_c; mode=ra_mode, thresh=ra_thresh, order=ra_order, accumulate=ra_acc)
	plot(plot(stft_c; title="Stft", colorbar=false, ylims=(0, ra_fmax)),
	     plot(reassigned; title="Reassign (mode=$ra_mode)", colorbar=false, ylims=(0, ra_fmax)); layout=(1, 2), size=(720, 300))
end

# ╔═╡ e6dffcf0-c8a1-4d22-b1c7-538791fb85bd
let
	F, Tm = get_reassigned(reassigned)
	S = get_spec(stft_c)
	keep = findall(S .> maximum(S) * 1e-3)
	scatter(Tm[keep], F[keep]; ms=1, msw=0, alpha=0.4, label="", ylims=(0, ra_fmax), xguide="Time (s)", yguide="Hz",
	        title="get_reassigned: where each STFT cell above −30 dB moved ($(length(keep)) cells)")
end

# ╔═╡ 0b6a480a-2280-4635-a164-d91a242f0c23
md"""
## 9. Cohen-class distributions — `Wvd` and `Cwd`

The Wigner-Ville distribution correlates the analytic signal with its time-reversed self: `z[n+m] conj(z[n-m])`, Fourier transformed over the lag `m`.
For a single chirp it is perfectly concentrated, but every pair of components produces an oscillating **interference term** halfway between them (try the *two tones* and the *chord*).
The Choi-Williams distribution smooths the lag products over time with an exponential kernel, trading some concentration for much weaker interference.

* `Wvd(frames)`: pseudo-WVD at every frame centre, lag window = the frame window. With `winsize = L` it has `L` bins at `(k-1)·sr/(2L)`: twice the frequency resolution of the STFT with the same window.
* `Cwd(frames; sigma=1, smooth=winsize ÷ 8)`: `sigma` the kernel parameter (smaller = more smoothing = weaker interference), `smooth` the half-length of the time smoothing in samples.
* `get_distribution(d)` returns the **signed** distribution; `get_spec` clips it at zero for the spectrogram interface.
* `wvd(x; nfft)` computes the full WVD of a short whole signal (cost `O(N · nfft)`).

Negative values (blue below) are the signature of interference.
"""

# ╔═╡ 7255e20d-f692-4eb6-b12d-bb918e84b419
md"""
winsize: $(@bind cw_win Select([64, 128, 256, 512]; default=256))
sigma: $(@bind cw_sigma Select([0.05, 0.1, 0.5, 1.0, 5.0, 20.0]; default=1.0))
smooth: $(@bind cw_smooth Slider(0:4:64; default=32, show_value=true))
max frequency shown (Hz): $(@bind cw_fmax Slider([1000, 2000, 4000, 8000]; default=2000, show_value=true))
"""

# ╔═╡ 19c551c0-ead8-4c67-84fc-27515fdbcfc6
begin
	cohen_frames = Frames(audio; winsize=cw_win, winstep, type=hanning)
	wvd_fr = Wvd(cohen_frames)
	cwd_fr = Cwd(cohen_frames; sigma=cw_sigma, smooth=cw_smooth)
end;

# ╔═╡ 7bad4a0b-c509-4d8f-95ac-5b62f9721d61
let
	function signed(d, title)
		D = get_distribution(d)
		m = maximum(abs, D)
		heatmap(get_times(d), get_freq(d), D ./ m; c=:balance, clims=(-1, 1), ylims=(0, cw_fmax),
		        title, xguide="Time (s)", colorbar=false)
	end
	plot(signed(wvd_fr, "Wvd (signed)"), signed(cwd_fr, "Cwd σ=$cw_sigma, smooth=$cw_smooth (signed)");
	     layout=(1, 2), size=(720, 320), yguide="Hz")
end

# ╔═╡ 42cabafb-42d1-4ccd-9f1b-a2afb83644ee
let x = vec(get_data(audio))[round(Int, 0.2sr)+1:round(Int, 0.2sr)+512]
	W = wvd(x; nfft=512)
	f = (0:size(W, 1)-1) .* (sr / (2 * 512))
	m = maximum(abs, W)
	heatmap(0.2 .+ (0:length(x)-1) ./ sr, f, W ./ m; c=:balance, clims=(-1, 1), ylims=(0, cw_fmax),
	        title="wvd(x): 512 samples from 0.2 s, every sample", xguide="Time (s)", yguide="Hz")
end

# ╔═╡ 37124a8b-edf1-42e1-a36c-4d37017eb55c
md"""
## 10. All front ends side by side

The same signal on the same time grid, through every front end with the settings chosen above (plus the plain `Stft` for reference).
With the *two tones* signal, look at which ones separate 440 and 470 Hz and which keep the clicks sharp; with the *chirp*, which ones follow the sweep as a thin line.
"""

# ╔═╡ bef96ef3-aff3-4dc6-95f2-79e38e0e6bbc
@bind cmp_log CheckBox(default=true)

# ╔═╡ afa08dc5-2d60-41b6-bad3-f1a0847bf6b4
md"logarithmic frequency axis (tick above)"

# ╔═╡ 8d145331-bf31-40ba-8867-6f5b8799fe40
let fs = cmp_log ? :log10 : :linear
	fst_fr = Fst(frames_rect; freqrange=(0, sr ÷ 2))
	items = [(Stft(frames_hann), "Stft"), (cwt_fr, "Cwt ($wname)"), (cqt, "Cqt"), (pwt_fr, "Pwt"),
	         (fst_fr, "Fst"), (nsgt_fr, "Nsgt"), (wsst_fr, "Wsst"), (reassigned, "Reassign"), (cwd_fr, "Cwd")]
	ps = [plot(s; freq_scale=fs, title=t, colorbar=false, xguide="", yguide="", titlefontsize=8, tickfontsize=6)
	      for (s, t) in items]
	plot(ps...; layout=(3, 3), size=(780, 640))
end

# ╔═╡ 5541e554-9ea9-48b4-a921-5c8e439adc7a
md"""
## 11. One interface, any front end

Every front end above implements `get_spec`, `get_freq`, `get_times`, `get_sr`: so the downstream stages do not care where the spectrogram came from.
Pick a front end: it feeds a `MelSpec` and then an `Mfcc` with no other change.
(The mel range is capped at the front end's highest frequency, since a filterbank cannot extend beyond the bins it receives.)
"""

# ╔═╡ 985f79bd-4f5e-4345-b667-ea8fecc9b0db
md"""
front end: $(@bind fe_name Select(["Stft", "Cwt", "Cqt", "Pwt", "Nsgt", "Wsst", "Synsq", "Reassign", "Cwd"]; default="Cqt"))
mel bands: $(@bind fe_nbands Slider(10:2:64; default=26, show_value=true))
"""

# ╔═╡ 1073cd36-1097-4f34-99d6-52366665af29
front_end = Dict("Stft" => stft_c, "Cwt" => cwt_fr, "Cqt" => cqt, "Pwt" => pwt_fr, "Nsgt" => nsgt_fr,
                 "Wsst" => wsst_fr, "Synsq" => synsq_fr, "Reassign" => reassigned, "Cwd" => cwd_fr)[fe_name]

# ╔═╡ 209337db-2f0e-42ac-a7e4-6624cfd8ff6b
mel_any = let f = get_freq(front_end)
	lo = max(50, ceil(Int, minimum(filter(>(0), f))))
	hi = min(floor(Int, maximum(f)) - 1, sr ÷ 2)
	MelSpec(front_end; nbands=fe_nbands, freqrange=(lo, hi))
end

# ╔═╡ 32e1b22a-29a2-459e-8a8a-c1ed0d19df4b
mfcc_any = Mfcc(mel_any; ncoeffs=13)

# ╔═╡ 70fdbd0d-fe55-4d38-9b9a-f5b57b5f3950
plot(plot(mel_any; title="MelSpec($fe_name)"), plot(mfcc_any; title="Mfcc(MelSpec($fe_name))"); layout=(2, 1), size=(720, 480))

# ╔═╡ 45dd60f1-624d-4544-8cc3-6da2025218a4
(front_end = typeof(front_end), front_end_bins = get_nbins(front_end), mel = size(get_data(mel_any)),
 mfcc = size(get_data(mfcc_any)), same_frames = get_times(mfcc_any) == get_times(front_end))

# ╔═╡ 1a96b33f-5a55-46bd-aa72-fe6aaa010b59
md"""
## Next

* **6 — Discrete wavelets and decompositions**: `Dwt`, `Wpt`, `Swt`, empirical mode decomposition and the Hilbert-Huang transform, the empirical wavelet transform.
* **7 — Music and rhythm**: chroma (also from the `Cqt` above), tonnetz, onsets, tempo, beats and harmonic/percussive separation.

The full tour: 01 loading and frames · 02 STFT and filterbanks · 03 cepstra · 04 spectral descriptors · **05 time-frequency** · 06 discrete wavelets and decompositions · 07 music and rhythm · 08 time domain and pitch · 09 signal processing and classic.
"""

# ╔═╡ Cell order:
# ╟─ffb26580-b763-4ce9-bf32-dea9a0e5f870
# ╟─9caa31f3-5baa-4845-9db2-739bf7135b72
# ╟─52ed1fb7-77f8-4cb4-836f-631378043e6e
# ╟─e0810837-579a-45f9-997d-260262790866
# ╟─76a120a4-3dec-40f6-8d01-36207edf35db
# ╠═80f572a1-e5a7-4799-b952-25bdcb75ced2
# ╠═cf94450c-2e04-4de2-bf67-9e5dff9ce04b
# ╠═42e8e4fc-790f-49ae-97b8-5936670545b3
# ╟─71baf269-918c-44cc-97e9-3275acf323dd
# ╟─7199a519-63d2-49e3-a326-9e4e38becd66
# ╠═eb0a1f65-d7c5-4fd9-86f4-35c5d0af351f
# ╠═ef1f470e-26d7-4b61-856c-3e9e5cc3cbdb
# ╠═a0ee1eb2-a0ff-43ce-89a5-ec0598024a00
# ╠═4df98457-9f16-4aad-8dbb-08b1e08cbb57
# ╠═40bb15b8-f619-4574-9373-2d00fc227bc7
# ╠═324f4360-63e5-4064-9395-69d8d99f6122
# ╟─255efd28-9a3e-4987-b396-861fe7b8e5bf
# ╠═c805020f-3077-4c90-b75d-cbf598502f05
# ╟─66473808-11bb-4688-9a78-72d6ea044433
# ╠═180cfe81-1d21-4520-b087-23e03609ea39
# ╠═aeff41ac-3223-4958-8e0a-32c7e0cc4b81
# ╠═3c81ccd1-7771-400b-bdce-f92d84f58577
# ╟─cf77b2a8-f7b9-4e78-9a7e-1cd424196709
# ╠═11ee93c2-de75-40b4-ab20-fa539b2130da
# ╠═a239f653-4508-419d-9d9a-eeaa44d18be8
# ╠═8227debd-b7c6-4974-bc27-024520c7194e
# ╠═de98263f-667b-4c9e-abf6-1ea97667324c
# ╟─5993953d-6eda-43de-b0db-93bff734f8c3
# ╟─fa8eced5-3a8e-4412-a9de-ac3233485597
# ╠═9ff7950a-9a15-4eeb-ae18-b32fa9278ad2
# ╠═6eb2d809-3e10-4e14-ae28-38cd53ca4567
# ╠═b60a913a-bf9c-42d8-91dd-b779291a4efa
# ╠═15e52fa2-00de-43be-885a-cc8179dfa3b8
# ╟─fcac67c2-eff6-4d99-883d-2ec50ae0d005
# ╠═8b6f570d-ccde-4995-86ec-a433e0362c31
# ╠═9f87bb80-41fd-46b6-af7a-9925d48b6ff9
# ╠═efce1f00-7cc0-4189-a1b6-2fb2e42f6329
# ╠═795ca5e9-7fb9-49d3-abe5-59c057d0dd47
# ╟─b41de932-a681-478a-888a-9973deff2bf1
# ╠═7087450f-db13-4530-8bbf-fdb39bbe60fa
# ╠═0232aa76-ccf0-4bf6-9793-9e2f1dd1cf55
# ╠═ebbe242c-2140-40a0-aacc-cf2d6da5a873
# ╠═a573bb5d-2067-4c6c-ae5d-628e3257fb1a
# ╠═42b1035e-084b-491f-baae-85c0702bc892
# ╟─28f6a6b6-9b40-497c-b287-bc53d4d750d5
# ╠═446c3ec8-8bdd-4bc1-a2dc-5eda0ee16495
# ╠═50e1a6ac-9ddd-492a-9b16-55ba82617275
# ╠═a829525c-72ee-46a6-b9a6-2e7582c99ba0
# ╟─899df21d-ad53-45e4-bd53-9f0bc2f0b906
# ╠═aa134e03-bdd7-4259-91d1-ef3c55378f51
# ╠═07531127-5c0f-428b-9f9c-4444cbb88444
# ╠═e6dffcf0-c8a1-4d22-b1c7-538791fb85bd
# ╟─0b6a480a-2280-4635-a164-d91a242f0c23
# ╠═7255e20d-f692-4eb6-b12d-bb918e84b419
# ╠═19c551c0-ead8-4c67-84fc-27515fdbcfc6
# ╠═7bad4a0b-c509-4d8f-95ac-5b62f9721d61
# ╠═42cabafb-42d1-4ccd-9f1b-a2afb83644ee
# ╟─37124a8b-edf1-42e1-a36c-4d37017eb55c
# ╠═bef96ef3-aff3-4dc6-95f2-79e38e0e6bbc
# ╟─afa08dc5-2d60-41b6-bad3-f1a0847bf6b4
# ╠═8d145331-bf31-40ba-8867-6f5b8799fe40
# ╟─5541e554-9ea9-48b4-a921-5c8e439adc7a
# ╠═985f79bd-4f5e-4345-b667-ea8fecc9b0db
# ╠═1073cd36-1097-4f34-99d6-52366665af29
# ╠═209337db-2f0e-42ac-a7e4-6624cfd8ff6b
# ╠═32e1b22a-29a2-459e-8a8a-c1ed0d19df4b
# ╠═70fdbd0d-fe55-4d38-9b9a-f5b57b5f3950
# ╠═45dd60f1-624d-4544-8cc3-6da2025218a4
# ╟─1a96b33f-5a55-46bd-aa72-fe6aaa010b59
