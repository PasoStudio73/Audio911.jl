# Helpers shared by the Audio911 demonstration notebooks.
# Each notebook includes this file once, right after activating the notebook
# environment, so the signal menu and the audio player look the same everywhere.

using Audio911
using Base64: base64encode

"Folder holding the audio files shipped with the tests."
const SAMPLES_DIR = joinpath(pkgdir(Audio911), "test", "test_files")

"""
    SIGNALS

Names offered by the signal menu of every notebook.
Speech comes from the test suite; the others are synthesised so that
resolution trade-offs are easy to read off a plot.
"""
const SIGNALS = [
    "speech"  => "Speech (test.wav, 16 kHz)",
    "chirp"   => "Log chirp 100 Hz → 6 kHz",
    "tones"   => "Two close tones (440 + 470 Hz) and clicks",
    "chord"   => "C major chord, then A minor",
]

"""
    demo_signal(name; sr=16000, duration=2.0) -> AudioFile

Load or synthesise the signal selected in the menu, always mono Float32.
"""
function demo_signal(name::AbstractString; sr::Int=16000, duration::Real=2.0)
    if name == "speech"
        return load(joinpath(SAMPLES_DIR, "test.wav"); sr, format=Float32, norm=true)
    elseif name == "chirp"
        x = chirp(100, 6000; sr, duration, T=Float32)
    elseif name == "tones"
        x = tone(440; sr, duration, T=Float32) .+ tone(470; sr, duration, T=Float32)
        c = clicks(0.25:0.5:duration-0.1; sr, click_freq=3000, click_duration=0.03,
                   length=length(x), T=Float32)
        x = x .* 0.4f0 .+ c
    elseif name == "chord"
        n = round(Int, duration * sr) ÷ 2
        cmaj = sum(tone(f; sr, duration=n / sr, T=Float32) for f in (261.63, 329.63, 392.0))
        amin = sum(tone(f; sr, duration=n / sr, T=Float32) for f in (220.0, 261.63, 329.63))
        x = vcat(cmaj, amin)
    else
        throw(ArgumentError("unknown demo signal $name"))
    end
    x = Float32.(x ./ maximum(abs, x))
    return AudioFile(x, sr)
end

"""
    audio_player(x, sr) / audio_player(audio)

An HTML `<audio>` element playing a mono signal, encoded as 16-bit WAV.
"""
function audio_player(x::AbstractVector{<:Real}, sr::Integer)
    peak = maximum(abs, x; init=0.0)
    y = peak > 0 ? x ./ peak : x
    pcm = round.(Int16, clamp.(y, -1, 1) .* 32767)
    io = IOBuffer()
    nbytes = 2 * length(pcm)
    write(io, b"RIFF", UInt32(36 + nbytes), b"WAVE", b"fmt ", UInt32(16), UInt16(1),
          UInt16(1), UInt32(sr), UInt32(2sr), UInt16(2), UInt16(16), b"data", UInt32(nbytes))
    write(io, htol.(pcm))
    src = "data:audio/wav;base64," * base64encode(take!(io))
    return HTML("<audio controls src=\"$src\" style=\"width:100%\"></audio>")
end
audio_player(a::AudioFile) = audio_player(vec(get_data(a)), get_sr(a))
