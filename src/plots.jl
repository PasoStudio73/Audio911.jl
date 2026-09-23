# ---------------------------------------------------------------------------- #
#                                Plots recipes                                 #
# ---------------------------------------------------------------------------- #
# Every stage of the pipeline can be handed to `Plots.plot`. The recipes are
# defined with RecipesBase, so Audio911 never loads Plots itself.

# spectrogram values in dB, respecting the spectrum kind
function _to_db(s::AbstractSpectrogram, top_db)
    S = get_spec(s)
    return get_spectrum(s) === magnitude ?
        amplitude_to_db(S; ref=maximum, amin=1e-5, top_db) :
        power_to_db(S; ref=maximum, amin=1e-10, top_db)
end

_freq_ticks(freq) = begin
    lo, hi = extrema(filter(>(0), freq))
    ticks = [10.0^k * m for k in floor(Int, log10(lo)):ceil(Int, log10(hi)) for m in (1, 2, 5)]
    ticks = filter(t -> lo ≤ t ≤ hi, ticks)
    (ticks, [t ≥ 1000 ? string(round(Int, t ÷ 1000), "k") : string(round(Int, t)) for t in ticks])
end

"""
    plot(audio::AudioFile)

Waveform, one series per channel, against time in seconds.
"""
@recipe function f(a::AudioFile)
    x = get_data(a)
    t = (0:size(x, 1)-1) ./ get_sr(a)
    xguide --> "Time (s)"
    yguide --> "Amplitude"
    title --> (isempty(get_path(a)) ? "Audio" : basename(get_path(a)))
    legend --> (size(x, 2) > 1)
    for c in 1:size(x, 2)
        @series begin
            label --> "channel $c"
            t, x[:, c]
        end
    end
end

"""
    plot(frames::Frames)

The framed signal as a waveform with the analysis window drawn on the
first frame, so the frame length and hop can be read off the plot.
"""
@recipe function f(fr::Frames)
    x  = get_signal(fr)
    sr = get_sr(fr)
    t  = ((0:length(x)-1) .+ get_offset(fr)) ./ sr
    xguide --> "Time (s)"
    yguide --> "Amplitude"
    title --> "Frames: $(length(fr)) × $(get_size(fr)) samples, hop $(get_step(fr))"
    @series begin
        label --> "signal"
        t, x
    end
    w = get_window(fr)
    s = fr.starts[1]
    @series begin
        label --> "window (frame 1)"
        linestyle --> :dash
        ((s-1:s+length(w)-2) .+ get_offset(fr)) ./ sr, w .* maximum(abs, x)
    end
end

"""
    plot(spec::AbstractSpectrogram; db=true, top_db=80, freq_scale=:linear)

Time-frequency heatmap of any spectrogram (`Stft`, `Cwt`, `LinSpec`,
`MelSpec`, `BarkSpec`, `ErbSpec`, derived spectrograms). Values are shown in
dB relative to the maximum, clipped `top_db` below it, unless `db=false`.
`freq_scale=:log10` gives a logarithmic frequency axis.
"""
@recipe function f(s::AbstractSpectrogram; db=true, top_db=80, freq_scale=:linear)
    z = db ? _to_db(s, top_db) : get_spec(s)
    freq = collect(get_freq(s))
    if freq_scale != :linear
        keep = findall(>(0), freq)
        freq = freq[keep]; z = z[keep, :]
        yscale --> freq_scale
        yticks --> _freq_ticks(freq)
    end
    seriestype := :heatmap
    xguide --> "Time (s)"
    yguide --> "Frequency (Hz)"
    colorbar_title --> (db ? "dB" : string(get_spectrum(s)))
    title --> string(nameof(typeof(s)))
    collect(get_times(s)), freq, z
end

"""
    plot(c::Cqt; db=true, top_db=80, freq_scale=:log10)

Constant-Q coefficients as a heatmap on a logarithmic frequency axis.
"""
@recipe function f(c::Cqt; db=true, top_db=80, freq_scale=:log10)
    z = db ? _to_db(c, top_db) : get_spec(c)
    freq = collect(get_freq(c))
    if freq_scale != :linear
        yscale --> freq_scale
        yticks --> _freq_ticks(freq)
    end
    seriestype := :heatmap
    xguide --> "Time (s)"
    yguide --> "Frequency (Hz)"
    colorbar_title --> (db ? "dB" : string(get_spectrum(c)))
    title --> "Cqt ($(get_setup(c).bins_per_octave)/octave)"
    collect(get_times(c)), freq, z
end

"""
    plot(s::Union{Pwt,Nsgt}; db=true, top_db=80, freq_scale=:log10)
    plot(s::Union{St,Fst}; db=true, top_db=80, freq_scale=:linear)

Heatmaps of the pooled whole-signal transforms; the scale-based ones
default to a logarithmic frequency axis.
"""
@recipe function f(s::Union{Pwt,Nsgt,St,Fst}; db=true, top_db=80,
                   freq_scale=(s isa Union{Pwt,Nsgt} ? :log10 : :linear))
    z = db ? _to_db(s, top_db) : get_spec(s)
    freq = collect(get_freq(s))
    if freq_scale != :linear
        keep = findall(>(0), freq)
        freq = freq[keep]; z = z[keep, :]
        yscale --> freq_scale
        yticks --> _freq_ticks(freq)
    end
    seriestype := :heatmap
    xguide --> "Time (s)"
    yguide --> "Frequency (Hz)"
    colorbar_title --> (db ? "dB" : string(get_spectrum(s)))
    title --> string(nameof(typeof(s)))
    collect(get_times(s)), freq, z
end

@recipe function f(c::Chroma)
    seriestype := :heatmap
    xguide --> "Time (s)"
    yguide --> "Pitch class"
    n = get_nbands(c)
    yticks --> (0:n-1, n == 12 ? collect(NOTE_NAMES) : string.(0:n-1))
    title --> "Chroma"
    collect(get_times(c)), collect(0:n-1), get_spec(c)
end

@recipe function f(t::Tonnetz)
    seriestype := :heatmap
    xguide --> "Time (s)"
    yguide --> "Tonal centroid"
    yticks --> (0:5, ["5th x", "5th y", "m3 x", "m3 y", "M3 x", "M3 y"])
    title --> "Tonnetz"
    collect(get_times(t)), collect(0:5), get_spec(t)
end

@recipe function f(t::Tempogram)
    seriestype := :heatmap
    xguide --> "Time (s)"
    yguide --> "Lag (frames)"
    title --> "Tempogram"
    collect(get_times(t)), collect(0:size(get_spec(t), 1)-1), get_spec(t)
end

@recipe function f(x::Union{SpectralContrast,PolyFeatures})
    seriestype := :heatmap
    xguide --> "Time (s)"
    yguide --> (x isa SpectralContrast ? "Band" : "Coefficient")
    title --> string(nameof(typeof(x)))
    collect(get_times(x)), collect(0:size(get_spec(x), 1)-1), get_spec(x)
end

"""
    plot(fbank::FBank)

Every filter of the bank against frequency.
"""
@recipe function f(fb::AbstractFBank)
    W = get_data(fb)
    xguide --> "Frequency (Hz)"
    yguide --> "Weight"
    legend --> false
    title --> "$(nameof(typeof(fb))): $(size(W, 1)) bands"
    # the grid the bank was evaluated on is stored for chroma banks; auditory
    # banks are drawn on a uniform grid of the same length
    x = fb isa ChromaFBank ? get_freq(fb) : range(0, get_sr(fb) / 2, length=size(W, 2))
    for k in 1:size(W, 1)
        @series begin
            x, W[k, :]
        end
    end
end

"""
    plot(c::AbstractCepstrum), plot(d::Delta)

Coefficients against time as a heatmap.
"""
@recipe function f(c::Union{AbstractCepstrum,AbstractDelta})
    seriestype := :heatmap
    xguide --> "Time (s)"
    yguide --> "Coefficient"
    title --> string(nameof(typeof(c)))
    S = get_spec(c)
    collect(get_times(c)), collect(0:size(S, 1)-1), S
end

"""
    plot(x::AbstractSpectral)

One-value-per-frame descriptors (spectral descriptors, `Rms`, `Zcr`,
`Pitch`, `OnsetStrength`, ...) as a line against time.
"""
@recipe function f(x::AbstractSpectral)
    xguide --> "Time (s)"
    yguide --> string(nameof(typeof(x)))
    label --> string(nameof(typeof(x)))
    collect(get_times(x)), get_data(x)
end

"""
    plot(h::Hpss)

Harmonic and percussive components side by side.
"""
@recipe function f(h::Hpss)
    layout := (2, 1)
    @series begin
        subplot := 1
        title --> "Harmonic"
        get_harmonic(h)
    end
    @series begin
        subplot := 2
        title --> "Percussive"
        get_percussive(h)
    end
end

"""
    plot(d::Deconv)

Timbre and pitch parts of a spectral deconvolution, one heatmap each.
"""
@recipe function f(d::Deconv)
    layout := (2, 1)
    t = collect(get_times(d)); fr = collect(get_freq(d))
    for (i, (name, M)) in enumerate((("Timbre", get_timbre(d)), ("Pitch", get_pitch(d))))
        @series begin
            subplot := i
            seriestype := :heatmap
            title --> name
            xguide --> "Time (s)"
            yguide --> "Bin frequency (Hz)"
            t, fr, M
        end
    end
end

"""
    plot(c::Cepstrogram)

Cepstrum (against quefrency), spectral envelope and details (against
frequency), one heatmap each.
"""
@recipe function f(c::Cepstrogram)
    layout := (3, 1)
    t = collect(get_times(c))
    @series begin
        subplot := 1
        seriestype := :heatmap
        title --> "Cepstrum"
        xguide --> "Time (s)"
        yguide --> "Quefrency (s)"
        t, get_quefrency(c), get_data(c)
    end
    for (i, (name, M)) in enumerate((("Envelope", get_envelope(c)), ("Details", get_details(c))))
        @series begin
            subplot := i + 1
            seriestype := :heatmap
            title --> name
            xguide --> "Time (s)"
            yguide --> "Frequency (Hz)"
            t, get_freq(c), M
        end
    end
end
