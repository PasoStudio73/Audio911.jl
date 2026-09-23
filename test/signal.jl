using Test
using Audio911

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)

@testset "conversions" begin
    @test hz_to_mel(1000) ≈ 15
    @test isapprox(hz_to_mel(1000; htk=true), 1000; rtol=1e-4)
    for f in (100.0, 1000.0, 4321.0), htk in (false, true)
        @test mel_to_hz(hz_to_mel(f; htk); htk) ≈ f
    end
    @test hz_to_midi(440) ≈ 69 && midi_to_hz(69) ≈ 440
    @test midi_to_note(69) == "A4" && midi_to_note(61; octave=false) == "C#"
    @test note_to_midi("A4") == 69 && note_to_midi("C#4") == 61 && note_to_midi("Bb3") == 58
    @test note_to_midi("C") == 12
    @test hz_to_note(261.63) == "C4" && note_to_hz("A4") ≈ 440
    @test midi_to_note(69.4; cents=true) == "A4+40"
    @test_throws ArgumentError note_to_midi("H2")
    @test collect(fft_frequencies(16000, 512)) ≈ collect(get_freq(Stft(rand(4096), 16000; winsize=512)))
    mf = mel_frequencies(40; fmin=0, fmax=8000)
    @test length(mf) == 40 && mf[1] == 0 && mf[end] ≈ 8000
    @test cqt_frequencies(13; fmin=55)[end] ≈ 110
    @test tempo_frequencies(4, 512, 22050)[2] ≈ 60 * 22050 / 512
    @test frames_to_samples(3, 256) == 513 && samples_to_frames(513, 256) == 3
    @test frames_to_time(3, 256, 16000) ≈ 0.032 && time_to_frames(0.032, 256, 16000) == 3
    @test samples_to_time(17, 16000) ≈ 0.001 && time_to_samples(0.001, 16000) == 17
    @test frames_to_samples(1, 256; offset=-256) == -255
end

@testset "decibels and weighting" begin
    @test power_to_db([1.0, 10.0, 100.0]; top_db=nothing) ≈ [0, 10, 20]
    @test power_to_db([1.0, 1e-20]; top_db=nothing) ≈ [0, -100]
    @test power_to_db([1.0, 1e-20]; top_db=30) ≈ [0, -30]
    @test power_to_db([1.0, 100.0]; ref=maximum, top_db=nothing) ≈ [-20, 0]
    @test amplitude_to_db([1.0, 10.0]; top_db=nothing) ≈ [0, 20]
    @test db_to_power(power_to_db([2.0, 5.0]; top_db=nothing)) ≈ [2, 5]
    @test db_to_amplitude(amplitude_to_db([2.0, 5.0]; top_db=nothing)) ≈ [2, 5]
    @test eltype(power_to_db(Float32[1, 2])) == Float32
    @test isapprox(A_weighting(1000), 0; atol=0.1)
    @test A_weighting(100) < -15
    @test A_weighting(1.0; min_db=-80) == -80
    @test isapprox(C_weighting(1000), 0; atol=0.1)
    S = fill(1.0, 3, 2); f = [100.0, 1000.0, 10000.0]
    @test perceptual_weighting(S, f; top_db=nothing)[:, 1] ≈ A_weighting.(f)
end

@testset "mu-law, normalize" begin
    x = range(-1, 1, length=201)
    q = mu_compress(x)
    @test eltype(q) <: Integer && extrema(q) == (-128, 127)
    @test maximum(abs.(mu_expand(q) .- x)) <= 0.021
    @test mu_expand(mu_compress(x; quantize=false); quantize=false) ≈ x
    @test normalize_signal([1.0, -2.0]) == [0.5, -1.0]
    @test normalize_signal([3.0, 4.0]; norm=2) ≈ [0.6, 0.8]
    @test normalize_signal([0.0, 0.0]) == [0.0, 0.0]
    M = [1.0 2.0; 3.0 8.0]
    @test normalize_signal(M; dims=1) ≈ [1/3 0.25; 1.0 1.0]
end

@testset "synthesis, trim, split, lpc" begin
    y = tone(440; sr=8000, duration=0.5)
    @test length(y) == 4000 && maximum(abs, y) ≈ 1
    @test isapprox(y[1], 0; atol=1e-12)
    c = chirp(100, 1000; sr=8000, duration=0.5)
    @test length(c) == 4000
    @test chirp(100, 1000; sr=8000, duration=0.5, linear=true) != c
    k = clicks([0.1, 0.3]; sr=8000)
    @test length(k) == round(Int, 0.3 * 8000) + 800
    @test k[802] != 0 && all(k[1:800] .== 0)
    @test length(clicks([0.1]; sr=8000, length=1000)) == 1000
    @test eltype(tone(440; sr=8000, duration=0.1, T=Float32)) == Float32

    sig = vcat(zeros(8000), 0.5 .* tone(440; sr=8000, duration=1), zeros(8000))
    tr, (a, b) = trim_silence(sig; frame_length=1024, hop_length=256)
    @test 7000 < a ≤ 8001 && 16000 ≤ b < 17100
    @test tr == sig[a:b]
    @test trim_silence(zeros(100))[2] == (1, 100)      # librosa keeps a silent signal whole
    iv = split_silence(vcat(sig, sig); frame_length=1024, hop_length=256)
    @test length(iv) == 2
    @test iv[1][1] ≤ 8001 ≤ iv[1][2] && iv[2][1] ≤ 8001 + 24000 ≤ iv[2][2]
    @test length(split_silence(zeros(5000))) == 1

    # Burg LPC recovers an AR(2) process
    a1, a2 = -1.2, 0.5
    n = 20000; e = randn(n); x = zeros(n)
    for i in 3:n; x[i] = -a1 * x[i-1] - a2 * x[i-2] + e[i]; end
    coeffs = lpc(x, 2)
    @test length(coeffs) == 3 && coeffs[1] == 1
    @test isapprox(coeffs[2], a1; atol=0.05) && isapprox(coeffs[3], a2; atol=0.05)
    @test_throws ArgumentError lpc(x, 0)
    @test eltype(lpc(Float32.(x), 2)) == Float32
end

@testset "file metadata" begin
    @test get_samplerate(test_file("test.wav")) == 16000
    @test get_samplerate(test_file("test.mp3")) == 44100
    @test get_samplerate(test_file("test.flac")) == 44100
    @test get_samplerate(test_file("test.ogg")) == 44100
    @test_throws ArgumentError get_samplerate(test_file("missing.wav"))
end
