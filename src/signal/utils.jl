# ---------------------------------------------------------------------------- #
#                               unit conversions                               #
# ---------------------------------------------------------------------------- #
"""
    hz_to_mel(f; htk=false)

Hz to mel: Slaney's formula (linear below 1 kHz, librosa default) or the
HTK formula `2595 log10(1 + f/700)`.
"""
function hz_to_mel(f::Real; htk::Bool=false)
    htk && return 2595 * log10(1 + f / 700)
    f_sp = 200 / 3
    f < 1000 && return f / f_sp
    return 1000 / f_sp + log(f / 1000) / (log(6.4) / 27)
end

"""
    mel_to_hz(m; htk=false)

Inverse of [`hz_to_mel`](@ref).
"""
function mel_to_hz(m::Real; htk::Bool=false)
    htk && return 700 * (10^(m / 2595) - 1)
    f_sp = 200 / 3
    min_log_mel = 1000 / f_sp
    m < min_log_mel && return m * f_sp
    return 1000 * exp((log(6.4) / 27) * (m - min_log_mel))
end

"""
    hz_to_midi(f)

MIDI note number of a frequency, `69 + 12 log2(f / 440)`.
"""
hz_to_midi(f::Real) = 12 * (log2(f) - log2(440)) + 69

"""
    midi_to_hz(n)

Frequency of a MIDI note number, `440 · 2^((n - 69) / 12)`.
"""
midi_to_hz(n::Real) = 440 * 2.0^((n - 69) / 12)

const NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

"""
    midi_to_note(n; octave=true, cents=false)

Note name of a MIDI number (`69 -> "A4"`); `cents=true` appends the
deviation from the nearest semitone.
"""
function midi_to_note(n::Real; octave::Bool=true, cents::Bool=false)
    k = round(Int, n)
    name = NOTE_NAMES[mod(k, 12) + 1]
    octave && (name *= string(k ÷ 12 - 1))
    cents && (name *= @sprintf("%+03d", round(Int, 100 * (n - k))))
    return name
end

"""
    note_to_midi(note::AbstractString)

MIDI number of a note name such as `"C4"`, `"F#3"`, `"Bb2"` or `"A"` (octave 0 when missing).
"""
function note_to_midi(note::AbstractString)
    m = match(r"^([A-Ga-g])([#♯b!♭]*)(-?\d+)?$", strip(note))
    isnothing(m) && throw(ArgumentError("cannot parse note name '$note'"))
    base = Dict('C' => 0, 'D' => 2, 'E' => 4, 'F' => 5, 'G' => 7, 'A' => 9, 'B' => 11)[uppercase(m[1][1])]
    acc = sum(c in ('#', '♯') ? 1 : -1 for c in m[2]; init=0)
    oct = isnothing(m[3]) ? 0 : parse(Int, m[3])
    return 12 * (oct + 1) + base + acc
end

"""
    hz_to_note(f; octave=true, cents=false)

Note name of a frequency (`440 -> "A4"`); see [`midi_to_note`](@ref).
"""
hz_to_note(f::Real; kwargs...) = midi_to_note(hz_to_midi(f); kwargs...)

"""
    note_to_hz(note)

Frequency of a note name (`"A4" -> 440`); see [`note_to_midi`](@ref).
"""
note_to_hz(note::AbstractString) = midi_to_hz(note_to_midi(note))

"""
    fft_frequencies(sr, nfft) -> AbstractRange

Centre frequencies of the one-sided FFT bins.
"""
fft_frequencies(sr::Int, nfft::Int) = (0:nfft÷2) .* (sr / nfft)

"""
    mel_frequencies(nmels=128; fmin=0, fmax=11025, htk=false) -> Vector

`nmels` frequencies equally spaced on the mel scale between `fmin` and `fmax`.
"""
mel_frequencies(nmels::Int=128; fmin::Real=0, fmax::Real=11025, htk::Bool=false) =
    [mel_to_hz(m; htk) for m in range(hz_to_mel(fmin; htk), hz_to_mel(fmax; htk), length=nmels)]

"""
    cqt_frequencies(n; fmin, bins_per_octave=12, tuning=0) -> Vector

Centre frequencies of `n` geometrically spaced bins starting at `fmin`.
"""
cqt_frequencies(n::Int; fmin::Real, bins_per_octave::Int=12, tuning::Real=0) =
    [fmin * 2.0^((k + tuning) / bins_per_octave) for k in 0:n-1]

"""
    frames_to_samples(i, hop; offset=0)

First sample (1-based) of frame `i` (1-based) with hop `hop`. `offset` is
the sample offset of frame 1 (see [`get_offset`](@ref)). Broadcasts over
arrays of indices.
"""
frames_to_samples(i, hop::Int; offset::Int=0) = @. (i - 1) * hop + offset + 1

"""
    samples_to_frames(n, hop; offset=0)

Frame (1-based) that starts at or before sample `n` (1-based); inverse of
[`frames_to_samples`](@ref).
"""
samples_to_frames(n, hop::Int; offset::Int=0) = @. fld(n - 1 - offset, hop) + 1

"""
    frames_to_time(i, hop, sr; offset=0)

Start time in seconds of frame `i`. Use [`get_times`](@ref) for frame centres.
"""
frames_to_time(i, hop::Int, sr::Int; offset::Int=0) = @. ((i - 1) * hop + offset) / sr

"""
    time_to_frames(t, hop, sr; offset=0)

Frame (1-based) that starts at or before time `t` in seconds.
"""
time_to_frames(t, hop::Int, sr::Int; offset::Int=0) = @. fld(round(Int, t * sr) - offset, hop) + 1

"""
    samples_to_time(n, sr)

Time in seconds of sample `n` (1-based).
"""
samples_to_time(n, sr::Int) = @. (n - 1) / sr

"""
    time_to_samples(t, sr)

Sample index (1-based) of time `t` in seconds.
"""
time_to_samples(t, sr::Int) = @. round(Int, t * sr) + 1

# ---------------------------------------------------------------------------- #
#                                   decibels                                   #
# ---------------------------------------------------------------------------- #
"""
    power_to_db(S; ref=1, amin=1e-10, top_db=80, min_db=nothing) -> Array

`10 log10(max(S, amin) / ref)`, clipped below `max - top_db` when `top_db`
is given (librosa `power_to_db`) and below `min_db` when it is given. `ref`
may be a number or a function of `S` such as `maximum`.

audioFlux's `power_to_db(X, min_db)` is `power_to_db(S; ref=maximum,
top_db=-min_db)`; its absolute `power_to_abs_db(X, fft_length)` is
`power_to_db(S; ref=fft_length^2, top_db=nothing, min_db=-80)` (and
`mag_to_abs_db` the same with [`amplitude_to_db`](@ref) and
`ref=fft_length`); their `is_norm` is `maximum(D) .- D`.
"""
function power_to_db(S::AbstractArray{T}; ref::Union{Real,Function}=1, amin::Real=1e-10,
                     top_db::Maybe{Real}=80, min_db::Maybe{Real}=nothing) where {T<:Real}
    r = ref isa Function ? T(ref(S)) : T(ref)
    a = T(amin)
    D = similar(S, T)
    @inbounds for i in eachindex(S)
        D[i] = 10 * log10(max(S[i], a)) - 10 * log10(max(r, a))
    end
    if !isnothing(top_db)
        lo = maximum(D) - T(top_db)
        @inbounds for i in eachindex(D)
            D[i] = max(D[i], lo)
        end
    end
    if !isnothing(min_db)
        md = T(min_db)
        @inbounds for i in eachindex(D)
            D[i] = max(D[i], md)
        end
    end
    return D
end

"""
    amplitude_to_db(S; ref=1, amin=1e-5, top_db=80, min_db=nothing) -> Array

`20 log10(max(S, amin) / ref)` with clipping (librosa `amplitude_to_db`).
"""
amplitude_to_db(S::AbstractArray{T}; ref::Union{Real,Function}=1, amin::Real=1e-5,
                top_db::Maybe{Real}=80, min_db::Maybe{Real}=nothing) where {T<:Real} =
    power_to_db(S .^ 2; ref=ref isa Function ? x -> ref(sqrt.(x))^2 : ref^2, amin=amin^2, top_db, min_db)

"""
    db_to_power(D; ref=1)

Inverse of [`power_to_db`](@ref): `ref · 10^(D/10)`.
"""
db_to_power(D::AbstractArray; ref::Real=1) = @. ref * 10^(D / 10)

"""
    db_to_amplitude(D; ref=1)

Inverse of [`amplitude_to_db`](@ref): `ref · 10^(D/20)`.
"""
db_to_amplitude(D::AbstractArray; ref::Real=1) = @. ref * 10^(D / 20)

# ---------------------------------------------------------------------------- #
#                              perceptual weighting                            #
# ---------------------------------------------------------------------------- #
"""
    A_weighting(f; min_db=-80)

IEC 61672 A-weighting curve in dB at frequency `f` (Hz), floored at `min_db`.
"""
function A_weighting(f::Real; min_db::Maybe{Real}=-80)
    f2 = float(f)^2
    num = 12194^2 * f2^2
    den = (f2 + 20.6^2) * sqrt((f2 + 107.7^2) * (f2 + 737.9^2)) * (f2 + 12194^2)
    w = 20 * log10(num / den) + 2.0
    return isnothing(min_db) ? w : max(w, min_db)
end

"""
    C_weighting(f; min_db=-80)

IEC 61672 C-weighting curve in dB at frequency `f` (Hz), floored at `min_db`.
"""
function C_weighting(f::Real; min_db::Maybe{Real}=-80)
    f2 = float(f)^2
    num = 12194^2 * f2
    den = (f2 + 20.6^2) * (f2 + 12194^2)
    w = 20 * log10(num / den) + 0.06
    return isnothing(min_db) ? w : max(w, min_db)
end

"""
    B_weighting(f; min_db=-80)

IEC 60651 B-weighting curve in dB at frequency `f` (Hz), floored at
`min_db` (audioFlux `auditory_weight_b`, librosa `B_weighting`).
"""
function B_weighting(f::Real; min_db::Maybe{Real}=-80)
    f2 = float(f)^2
    w = 0.17 + 20 * (log10(12194.0^2) + 1.5 * log10(f2) - log10(f2 + 12194.0^2) -
                     log10(f2 + 20.6^2) - 0.5 * log10(f2 + 158.5^2))
    return isnothing(min_db) ? w : max(w, min_db)
end

"""
    D_weighting(f; min_db=-80)

IEC 537 D-weighting curve in dB at frequency `f` (Hz), floored at `min_db`
(librosa `D_weighting`). audioFlux's `auditory_weight_d` has a typo in the
second pole pair (`(3136.5² - f²)(1018.7² - f²)` for `(3136.5² - f²)²`),
which lifts its curve by up to 9.8 dB (below 1 kHz); this is the standard
curve.
"""
function D_weighting(f::Real; min_db::Maybe{Real}=-80)
    f2 = float(f)^2
    w = 20 * (0.5 * log10(f2) - log10(8.3046305e-3^2) +
              0.5 * (log10((1018.7^2 - f2)^2 + 1039.6^2 * f2) - log10((3136.5^2 - f2)^2 + 3424.0^2 * f2) -
                     log10(282.7^2 + f2) - log10(1160.0^2 + f2)))
    return isnothing(min_db) ? w : max(w, min_db)
end

"""
    perceptual_weighting(S, freq; weighting=A_weighting, kwargs...) -> Array

Power spectrogram in dB with a frequency weighting curve added to every bin
(librosa `perceptual_weighting`).
"""
perceptual_weighting(S::AbstractMatrix, freq::AbstractVector; weighting::Function=A_weighting, kwargs...) =
    power_to_db(S; kwargs...) .+ [weighting(f) for f in freq]

# ---------------------------------------------------------------------------- #
#                                    mu-law                                    #
# ---------------------------------------------------------------------------- #
"""
    mu_compress(x; mu=255, quantize=true)

μ-law compression of a signal in `[-1, 1]` (librosa `mu_compress`). With
`quantize=true` the values are integers in `-(mu+1)/2 : (mu-1)/2`.
"""
function mu_compress(x::AbstractArray{T}; mu::Real=255, quantize::Bool=true) where {T<:Real}
    y = @. sign(x) * log1p(mu * abs(x)) / log1p(mu)
    quantize || return y
    return @. floor(Int, (y + 1) / 2 * mu + 0.5) - (mu + 1) ÷ 2
end

"""
    mu_expand(x; mu=255, quantize=true)

Inverse of [`mu_compress`](@ref).
"""
function mu_expand(x::AbstractArray; mu::Real=255, quantize::Bool=true)
    y = quantize ? (@. (x + (mu + 1) ÷ 2) * 2 / mu - 1) : x
    return @. sign(y) * ((1 + mu)^abs(y) - 1) / mu
end

"""
    normalize_signal(x; norm=Inf, dims=:) -> Array

Scale `x` by its `norm`-norm (`Inf`: maximum absolute value, `1`, `2`) over
the whole array or along `dims` (librosa `util.normalize`). Zero norms are left alone. Named to avoid the
clash with `LinearAlgebra.normalize`.
"""
function normalize_signal(x::AbstractArray{T}; norm::Real=Inf, dims=:) where {T<:Real}
    if dims === Colon()
        n = isinf(norm) ? maximum(abs, x) : sum(abs.(x) .^ norm)^(1 / norm)
        return n > 0 ? x ./ n : copy(x)
    end
    n = isinf(norm) ? maximum(abs, x; dims) : sum(abs.(x) .^ norm; dims) .^ (1 / norm)
    return x ./ ifelse.(n .> 0, n, one(T))
end

# ---------------------------------------------------------------------------- #
#                                   synthesis                                  #
# ---------------------------------------------------------------------------- #
"""
    tone(frequency; sr, duration, phi=-π/2, T=Float64) -> Vector

Sinusoid `cos(2π f t + phi)` of `duration` seconds (librosa `tone`).
"""
function tone(frequency::Real; sr::Int, duration::Real, phi::Real=-π/2, T::Type=Float64)
    n = round(Int, duration * sr)
    return T[cos(2π * frequency * k / sr + phi) for k in 0:n-1]
end

"""
    chirp(fmin, fmax; sr, duration, linear=false, phi=-π/2, T=Float64) -> Vector

Frequency sweep from `fmin` to `fmax` over `duration` seconds, exponential
by default or `linear=true` (librosa `chirp`).
"""
function chirp(fmin::Real, fmax::Real; sr::Int, duration::Real, linear::Bool=false, phi::Real=-π/2, T::Type=Float64)
    n = round(Int, duration * sr)
    y = Vector{T}(undef, n)
    for k in 0:n-1
        t = k / sr
        φ = linear ? 2π * (fmin * t + (fmax - fmin) / (2duration) * t^2) :
                     2π * fmin * duration / log(fmax / fmin) * ((fmax / fmin)^(t / duration) - 1)
        y[k + 1] = cos(φ + phi)
    end
    return y
end

"""
    clicks(times; sr, click_freq=1000, click_duration=0.1, length=nothing, T=Float64) -> Vector

Signal with an exponentially decaying click at every time in `times`
seconds (librosa `clicks`), of `length` samples (default: covering the last click).
"""
function clicks(times::AbstractVector{<:Real}; sr::Int, click_freq::Real=1000, click_duration::Real=0.1,
                length::Maybe{Int}=nothing, T::Type=Float64)
    nc = round(Int, click_duration * sr)
    click = T[sin(2π * click_freq * k / sr) * exp(-10k / nc) for k in 0:nc-1]
    n = isnothing(length) ? (isempty(times) ? nc : round(Int, maximum(times) * sr) + nc) : length
    y = zeros(T, n)
    for t in times
        s = round(Int, t * sr) + 1
        for k in 1:nc
            i = s + k - 1
            i ≤ n && (y[i] += click[k])
        end
    end
    return y
end

# ---------------------------------------------------------------------------- #
#                                  trim / split                                #
# ---------------------------------------------------------------------------- #
# frame-wise dB level of the signal relative to its peak
function _frame_db(x::AbstractVector{T}, frame_length::Int, hop::Int) where T
    fr = Frames(x, 1; winsize=frame_length, winstep=hop, type=rect, center=true)
    r  = get_data(Rms(fr))
    return power_to_db(r .^ 2; ref=maximum, top_db=nothing), fr
end

"""
    trim_silence(x; top_db=60, frame_length=2048, hop_length=512) -> (y, (first, last))

Remove leading and trailing silence, where silence is any frame more than
`top_db` below the loudest frame (librosa `effects.trim`; a silent signal is kept
whole, as in librosa). Returns the
trimmed signal and the 1-based sample interval kept.
"""
function trim_silence(x::AbstractVector{T}; top_db::Real=60, frame_length::Int=2048, hop_length::Int=512) where {T<:Real}
    db, fr = _frame_db(Vector{float(T)}(x), frame_length, hop_length)
    keep = findall(>(-top_db), db)
    isempty(keep) && return (x[1:0], (1, 0))
    first = (keep[1] - 1) * hop_length + 1
    last  = min(length(x), keep[end] * hop_length)
    return (x[first:last], (first, last))
end

"""
    split_silence(x; top_db=60, frame_length=2048, hop_length=512) -> Vector{Tuple{Int,Int}}

Sample intervals (1-based, inclusive) of the non-silent parts of `x`
(librosa `effects.split`).
"""
function split_silence(x::AbstractVector{T}; top_db::Real=60, frame_length::Int=2048, hop_length::Int=512) where {T<:Real}
    db, fr = _frame_db(Vector{float(T)}(x), frame_length, hop_length)
    loud = db .> -top_db
    intervals = Tuple{Int,Int}[]
    n = length(loud); i = 1
    while i ≤ n
        if loud[i]
            j = i
            while j < n && loud[j + 1]; j += 1; end
            first = (i - 1) * hop_length + 1
            last  = min(length(x), j * hop_length)
            push!(intervals, (first, last))
            i = j + 1
        else
            i += 1
        end
    end
    return intervals
end

# ---------------------------------------------------------------------------- #
#                                      lpc                                     #
# ---------------------------------------------------------------------------- #
"""
    lpc(x, order) -> Vector

Linear prediction coefficients `[1, a1, ..., a_order]` by Burg's method
(librosa `lpc`).
"""
function lpc(x::AbstractVector{T}, order::Int) where {T<:Real}
    F = float(T)
    n = length(x)
    order ≥ 1 || throw(ArgumentError("order must be ≥ 1"))
    n > order || throw(ArgumentError("signal shorter than the LPC order"))
    f = Vector{F}(x); b = copy(f)
    a = zeros(F, order + 1); a[1] = 1
    ar = zeros(F, order + 1)
    den = 2 * sum(abs2, f)
    for i in 1:order
        num = zero(F)
        for j in i+1:n
            num += f[j] * b[j - 1]
        end
        den -= f[i]^2 + b[n]^2   # remove the samples leaving the window
        den > 0 || break
        k = -2num / den
        ar .= a
        for j in 1:i
            a[j + 1] = ar[j + 1] + k * ar[i - j + 1]
        end
        for j in n:-1:i+1
            fj = f[j]
            f[j] = fj + k * b[j - 1]
            b[j] = b[j - 1] + k * fj
        end
        den = (1 - k^2) * den
    end
    return a
end

# ---------------------------------------------------------------------------- #
#                                 file metadata                                #
# ---------------------------------------------------------------------------- #
"""
    get_samplerate(path) -> Int

Sample rate stored in an audio file's header, without decoding the audio.
"""
function get_samplerate(path::AbstractString)
    sym = detect_format(path)
    if sym == :MP3
        mh = _mpg123_new()
        try
            _mpg123_open(mh, String(path))
            try
                return _mpg123_getformat(mh)[1]
            finally
                _mpg123_close(mh)
            end
        finally
            _mpg123_delete(mh)
        end
    else
        info = SF_INFO()
        ptr  = _sf_open(String(path), info)
        _sf_close(ptr)
        return Int(info.samplerate)
    end
end

# ---------------------------------------------------------------------------- #
#                          audioFlux feature utilities                         #
# ---------------------------------------------------------------------------- #
# audioFlux utils/scale.py, convert.py (temproal_db) and util.py (synth_f0)
# (MIT licence, Copyright (c) 2023 libAudioFlux).

# quartile of a sorted vector at the 1-based position `p` (audioFlux's rule:
# the element when p is whole, else the mean of its two neighbours)
_quartile(s::AbstractVector, num::Int, den::Int) = begin
    n = length(s)
    i = (n + 1) * num ÷ den
    (n + 1) * num % den == 0 ? s[max(i, 1)] : (s[max(i, 1)] + s[min(i + 1, n)]) / 2
end

function _scale_vec!(y::AbstractVector{T}, x::AbstractVector, method::Symbol, corrected::Bool) where T
    if method === :minmax
        lo, hi = extrema(x)
        hi > lo ? (y .= (x .- lo) ./ (hi - lo)) : fill!(y, zero(T))
    elseif method === :standard
        μ = mean(x); σ = std(x; corrected)
        σ > 0 ? (y .= (x .- μ) ./ σ) : fill!(y, zero(T))
    elseif method === :maxabs
        m = maximum(abs, x)
        m > 0 ? (y .= x ./ m) : fill!(y, zero(T))
    elseif method === :robust
        s = sort(x)
        q1, q2, q3 = _quartile(s, 1, 4), _quartile(s, 1, 2), _quartile(s, 3, 4)
        q3 > q1 ? (y .= (x .- q2) ./ (q3 - q1)) : fill!(y, zero(T))
    elseif method === :center
        y .= x .- mean(x)
    elseif method === :mean
        lo, hi = extrema(x)
        hi > lo ? (y .= (x .- mean(x)) ./ (hi - lo)) : fill!(y, zero(T))
    elseif method === :arctan
        y .= atan.(x) ./ (T(π) / 2)
    else
        throw(ArgumentError("method must be :minmax, :standard, :maxabs, :robust, :center, :mean or :arctan, got :$method"))
    end
    return y
end

"""
    feature_scale(X; method=:minmax, dims=1, corrected=false) -> Array

Scale every column (`dims=1`, audioFlux's samples × features layout) or
row (`dims=2`) of a matrix, or a vector, with one of audioFlux's feature
scalers (`utils.scale`):

| `method` | result | audioFlux |
|:---------|:-------|:----------|
| `:minmax` | `(x - min) / (max - min)` | `min_max_scale` |
| `:standard` | `(x - mean) / std` (`corrected=true` for the sample std) | `stand_scale` |
| `:maxabs` | `x / max(abs(x))` | `max_abs_scale` |
| `:robust` | `(x - median) / (Q3 - Q1)` | `robust_scale` |
| `:center` | `x - mean` | `center_scale` |
| `:mean` | `(x - mean) / (max - min)` | `mean_scale` |
| `:arctan` | `atan(x) / (π/2)` | `arctan_scale` |

A constant column gives zeros. The quartiles take the element at position
`(n+1)/4` (or the mean of the two around it) of the sorted column;
audioFlux reads them from the unsorted column, which only agrees for
sorted data.
"""
function feature_scale(X::AbstractArray{<:Real}; method::Symbol=:minmax, dims::Int=1, corrected::Bool=false)
    T = float(eltype(X))
    Y = similar(X, T)
    if X isa AbstractVector
        _scale_vec!(Y, X, method, corrected)
    else
        dims in (1, 2) || throw(ArgumentError("dims must be 1 or 2, got $dims"))
        for (y, x) in zip(eachslice(Y; dims=3 - dims), eachslice(X; dims=3 - dims))
            _scale_vec!(y, x, method, corrected)
        end
    end
    return Y
end

"""
    temporal_db(x; base=18) -> (max_db, mean_db, quiet)

Level summary of a signal (audioFlux `temproal_db`): the sample levels
`20 log10(|x| + 10⁻⁸)`, floored at -36 dB, give their maximum and mean, and
`quiet` is the fraction of samples at or below `-base` dB.
"""
function temporal_db(x::AbstractVector{<:Real}; base::Real=18)
    isempty(x) && throw(ArgumentError("x is empty"))
    T = float(eltype(x))
    v = T[max(20 * log10(abs(s) + T(1e-8)), T(-36)) for s in x]
    return maximum(v), mean(v), count(≤(-base), v) / length(v)
end

"""
    synth_f0(times, frequencies, sr; amplitudes=nothing) -> Vector

Synthesise the sinusoid of a pitch curve (audioFlux `synth_f0`): the
frequencies (Hz) and amplitudes given at `times` (seconds) are linearly
interpolated at every sample up to `floor(times[end] · sr)` (extrapolated
linearly before `times[1]`, held after the end), and the phase is the
running sum of `2π f / sr`. Amplitudes default to 1.
"""
function synth_f0(times::AbstractVector{<:Real}, frequencies::AbstractVector{<:Real}, sr::Int;
                  amplitudes::Maybe{AbstractVector{<:Real}}=nothing)
    n = length(times)
    n == length(frequencies) || throw(DimensionMismatch("times and frequencies must have the same length"))
    isnothing(amplitudes) || length(amplitudes) == n ||
        throw(DimensionMismatch("amplitudes must have the length of times"))
    n ≥ 1 || throw(ArgumentError("times is empty"))
    T = float(promote_type(eltype(times), eltype(frequencies)))
    N = floor(Int, times[end] * sr)
    ts = T.(times) .* sr
    ω = T.(frequencies) .* (2T(π) / sr)
    function interp(vals, t, k)
        while k < n && t > ts[k + 1]
            k += 1
        end
        v = k < n ? vals[k] + (t - ts[k]) * (vals[k + 1] - vals[k]) / (ts[k + 1] - ts[k]) : vals[n]
        return v, k
    end
    y = Vector{T}(undef, N)
    amp = isnothing(amplitudes) ? nothing : T.(amplitudes)
    kf = 1; ka = 1; φ = zero(T)
    for i in 0:N-1
        w, kf = interp(ω, T(i), kf)
        φ += w
        a = one(T)
        isnothing(amp) || ((a, ka) = interp(amp, T(i), ka))
        y[i + 1] = sin(φ) * a
    end
    return y
end
