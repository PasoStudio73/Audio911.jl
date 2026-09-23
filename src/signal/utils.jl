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
    hz_to_midi(f), midi_to_hz(n)

MIDI note number of a frequency (`69 + 12 log2(f/440)`) and back.
"""
hz_to_midi(f::Real) = 12 * (log2(f) - log2(440)) + 69
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
    hz_to_note(f; kwargs...), note_to_hz(note)

Note name of a frequency and frequency of a note name.
"""
hz_to_note(f::Real; kwargs...) = midi_to_note(hz_to_midi(f); kwargs...)
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
    frames_to_samples(i, hop; offset=0), samples_to_frames(n, hop; offset=0)
    frames_to_time(i, hop, sr; offset=0), time_to_frames(t, hop, sr; offset=0)
    samples_to_time(n, sr), time_to_samples(t, sr)

Conversions between frame indices (1-based), sample indices (1-based) and
seconds. `offset` is the sample at which frame 1 starts (see
[`get_offset`](@ref)); frame times are frame starts, use [`get_times`](@ref)
for centres.
"""
frames_to_samples(i, hop::Int; offset::Int=0) = @. (i - 1) * hop + offset + 1
samples_to_frames(n, hop::Int; offset::Int=0) = @. fld(n - 1 - offset, hop) + 1
frames_to_time(i, hop::Int, sr::Int; offset::Int=0) = @. ((i - 1) * hop + offset) / sr
time_to_frames(t, hop::Int, sr::Int; offset::Int=0) = @. fld(round(Int, t * sr) - offset, hop) + 1
samples_to_time(n, sr::Int) = @. (n - 1) / sr
time_to_samples(t, sr::Int) = @. round(Int, t * sr) + 1

# ---------------------------------------------------------------------------- #
#                                   decibels                                   #
# ---------------------------------------------------------------------------- #
"""
    power_to_db(S; ref=1, amin=1e-10, top_db=80) -> Array

`10 log10(max(S, amin) / ref)`, clipped below `max - top_db` when `top_db`
is given (librosa `power_to_db`). `ref` may be a number or a function of
`S` such as `maximum`.
"""
function power_to_db(S::AbstractArray{T}; ref::Union{Real,Function}=1, amin::Real=1e-10,
                     top_db::Maybe{Real}=80) where {T<:Real}
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
    return D
end

"""
    amplitude_to_db(S; ref=1, amin=1e-5, top_db=80) -> Array

`20 log10(max(S, amin) / ref)` with clipping (librosa `amplitude_to_db`).
"""
amplitude_to_db(S::AbstractArray{T}; ref::Union{Real,Function}=1, amin::Real=1e-5,
                top_db::Maybe{Real}=80) where {T<:Real} =
    power_to_db(S .^ 2; ref=ref isa Function ? x -> ref(sqrt.(x))^2 : ref^2, amin=amin^2, top_db)

"""
    db_to_power(D; ref=1), db_to_amplitude(D; ref=1)

Inverses of [`power_to_db`](@ref) and [`amplitude_to_db`](@ref).
"""
db_to_power(D::AbstractArray; ref::Real=1) = @. ref * 10^(D / 10)
db_to_amplitude(D::AbstractArray; ref::Real=1) = @. ref * 10^(D / 20)

# ---------------------------------------------------------------------------- #
#                              perceptual weighting                            #
# ---------------------------------------------------------------------------- #
"""
    A_weighting(f; min_db=-80), C_weighting(f; min_db=-80)

IEC 61672 A- and C-weighting curves in dB at frequency `f` (Hz), floored at `min_db`.
"""
function A_weighting(f::Real; min_db::Maybe{Real}=-80)
    f2 = float(f)^2
    num = 12194^2 * f2^2
    den = (f2 + 20.6^2) * sqrt((f2 + 107.7^2) * (f2 + 737.9^2)) * (f2 + 12194^2)
    w = 20 * log10(num / den) + 2.0
    return isnothing(min_db) ? w : max(w, min_db)
end

function C_weighting(f::Real; min_db::Maybe{Real}=-80)
    f2 = float(f)^2
    num = 12194^2 * f2
    den = (f2 + 20.6^2) * (f2 + 12194^2)
    w = 20 * log10(num / den) + 0.06
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
    mu_compress(x; mu=255, quantize=true), mu_expand(x; mu=255, quantize=true)

μ-law companding of a signal in `[-1, 1]` (librosa `mu_compress`/`mu_expand`).
With `quantize=true` the compressed values are integers in `-(mu+1)/2 : (mu-1)/2`.
"""
function mu_compress(x::AbstractArray{T}; mu::Real=255, quantize::Bool=true) where {T<:Real}
    y = @. sign(x) * log1p(mu * abs(x)) / log1p(mu)
    quantize || return y
    return @. floor(Int, (y + 1) / 2 * mu + 0.5) - (mu + 1) ÷ 2
end

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
