using Test
using Audio911
using MAT

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
af_file(filename)   = joinpath(dirname(@__FILE__), "audioflux_files", "pitch", filename)
wav_file = test_file("test.wav")
semis(a, b) = abs(12 * log2(max(a, 1e-9) / max(b, 1e-9)))

@testset "spectral pitch estimators" begin
    sr = 16000
    t  = (0:sr-1) ./ sr
    for f0 in (98.0, 196.0, 330.0)
        x = sum(sin.(2π * k * f0 .* t) ./ k for k in 1:6)
        fr = Frames(x, sr; winsize=1024, winstep=512)
        for meth in (pitch_pef, pitch_hps, pitch_lhs, pitch_stft)
            p = get_data(Pitch(fr; method=meth, range=(50, 500)))
            @test all(semis.(p, f0) .< 0.5)          # PEF's log grid biases it by up to ~0.4 st
        end
    end
    # a missing fundamental is recovered from its harmonics
    x = sum(sin.(2π * k * 150 .* t) for k in 2:6)
    fr = Frames(x, sr; winsize=2048, winstep=1024)
    for meth in (pitch_hps, pitch_lhs, pitch_pef, pitch_stft)
        @test all(semis.(get_data(Pitch(fr; method=meth, range=(80, 400))), 150) .< 0.5)
    end
    # options and types
    x32 = Float32.(sin.(2π * 220 .* t))
    f32 = Frames(x32, sr; winsize=1024, winstep=512)
    for meth in (pitch_pef, pitch_hps, pitch_lhs, pitch_stft)
        @test eltype(get_data(Pitch(f32; method=meth))) == Float32
    end
    h32 = Float32.(sum(sin.(2π * k * 220 .* t) ./ k for k in 1:4))
    @test pitch_hps(h32[1:1024], sr; harmonics=3, window=hanning) ≈ 220 atol=2
    @test pitch_stft(zeros(1024), sr) == 0
    @test pitch_pef(x32[1:1024], sr; alpha=5, beta=0.6, gamma=1.6) ≈ 220 rtol=0.03
end

# audioFlux calls: test/audioflux_sources/pitch.py (frames of 1024, hop 256,
# range 50-500 Hz) on test.wav and on a harmonic sweep 110 → 220 Hz.
@testset "pitch against audioFlux" begin
    m  = MAT.matread(af_file("pitch.mat"))
    sr = 16000
    audio = Audio911.load(wav_file; format=Float64)
    tone  = Float64.(vec(m["tone"]))
    for (name, sig) in (("wav", vec(get_data(audio))), ("tone", tone))
        fr = Frames(sig, sr; winsize=1024, winstep=256)
        # af.PitchPEF(... cut_fre=4000, window_type=HAMM, alpha=10, beta=0.5, gamma=1.8): identical
        # (a near-tie between two log-grid bins in an unvoiced frame can go
        # either way in float32: at most two frames, one grid step apart)
        p, r = get_data(Pitch(fr; method=pitch_pef, range=(50, 500))), vec(m["pef_$name"])
        @test count(abs.(p .- r) ./ r .> 1e-4) ≤ 2
        @test maximum(abs.(p .- r) ./ r) < 0.005
        # af.PitchHPS / PitchLHS(harmonic_count=5, HAMM): audioFlux searches the
        # bins 50…500 and reports (bin + 1)·sr/N; on the same bins (range
        # 48-489 Hz here) the two differ by exactly one bin
        for (meth, key) in ((pitch_hps, "hps"), (pitch_lhs, "lhs"))
            p = get_data(Pitch(fr; method=meth, range=(48, 489)))
            @test p .+ sr / 16384 ≈ vec(m["$(key)_$name"]) rtol=1e-5
        end
    end
    # the time-domain estimators already in Audio911 agree with audioFlux's
    # on a voiced signal (on the mostly unvoiced test.wav audioFlux's YIN
    # returns 0 and its NCF/CEP sit on the range edge, so there is nothing to compare)
    fr = Frames(tone, sr; winsize=1024, winstep=256)
    for (meth, key) in ((pitch_ncf, "ncf"), (pitch_cep, "cep"), (pitch_yin, "yin"))
        p = get_data(Pitch(fr; method=meth, range=(50, 500)))
        ref = vec(m["$(key)_tone"])
        @test count(semis.(p, ref) .< 0.5) / length(ref) ≥ 0.95
    end
    # audioFlux 0.1.9's PitchSTFT misses the fundamental of the clean sweep
    # (its bounds are swapped in pitchSTFTObj_new); Audio911's harmonic sieve
    # is checked against the known sweep instead
    tc = (collect(fr.starts) .- 1 .+ 512) ./ sr
    truth = 110 .* 2 .^ (tc ./ 2)
    @test all(semis.(get_data(Pitch(fr; method=pitch_stft, range=(50, 500))), truth) .< 0.5)
    ref = vec(m["stft_tone"])
    @test count(semis.(ref[1:length(truth)], truth) .< 0.5) / length(truth) < 0.5
end
