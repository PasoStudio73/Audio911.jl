using Test
using Audio911
using Statistics: mean

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
wav_file = test_file("test.wav")

sr = 16000
t  = (0:2sr-1) ./ sr
tone220 = sin.(2π * 220 .* t)
noise   = 0.3 .* (2 .* rand(length(t)) .- 1)

@testset "time-domain descriptors" begin
    fr = Frames(tone220, sr; winsize=1024, winstep=512, type=rect)
    n  = length(fr)
    r = Rms(fr)
    @test length(get_data(r)) == n
    @test all(isapprox.(get_data(r), 1 / sqrt(2); atol=0.02))
    @test get_times(r) == get_times(Stft(fr))
    rs = Rms(Stft(fr))
    @test isapprox(get_data(rs), get_data(r); rtol=0.05)
    e = Energy(fr; windowed=false)
    @test get_data(e) ≈ get_energy(fr)
    @test all(get_data(Energy(fr)) .<= get_data(e) .+ 1e-9)
    z = Zcr(fr)
    @test all(isapprox.(get_data(z), 2 * 220 / sr; atol=0.003))
    @test get_data(Zcr(fr; rate=false)) ≈ get_data(z) .* 1024
    zc = zero_crossings([1.0, -1.0, -1.0, 1.0, 0.0, 1.0])
    @test zc == [false, true, false, true, false, false]
    ac = autocorrelate(tone220[1:4096]; max_size=200)
    @test length(ac) == 200 && argmax(ac) == 1
    @test isapprox(ac[1], sum(abs2, tone220[1:4096]); rtol=1e-6)
    @test eltype(get_data(Rms(Frames(Float32.(tone220), sr; winsize=512)))) == Float32
end

@testset "pitch and harmonic ratio" begin
    fr = Frames(tone220, sr; winsize=2048, winstep=512, type=rect)
    for m in (pitch_ncf, pitch_yin, pitch_cep)
        p = Pitch(fr; method=m, range=(60, 500))
        f0 = get_data(p)
        @test length(f0) == length(fr)
        @test all(isapprox.(f0, 220; rtol=0.03))
        @test get_setup(p).method === m
    end
    @test isapprox(pitch_yin(tone220[1:2048], sr; range=(60, 500)), 220; rtol=0.02)
    fn = Frames(noise, sr; winsize=2048, winstep=512, type=rect)
    hr_tone  = get_data(HarmonicRatio(fr))
    hr_noise = get_data(HarmonicRatio(fn))
    @test all(hr_tone .> 0.95)
    @test mean(hr_noise) < 0.5
    @test_throws ArgumentError Pitch(Frames(tone220, sr; winsize=32); range=(50, 400))
    # silence is unvoiced
    fs = Frames(zeros(8000), sr; winsize=1024, winstep=512)
    @test all(get_data(Pitch(fs)) .== 0)
end

@testset "chroma, tonnetz, contrast, poly" begin
    a440 = sin.(2π * 440 .* t)
    stft = Stft(a440, sr; winsize=2048, winstep=512, type=hanning)
    fb = chroma_fbank(stft)
    @test size(get_data(fb)) == (12, get_nbins(stft))
    @test get_nbands(fb) == 12
    c = Chroma(stft)
    @test size(get_spec(c)) == (12, get_nframes(stft))
    @test get_freq(c) == 0:11
    @test all(argmax(view(get_spec(c), :, j)) == 10 for j in 1:get_nframes(c))   # A = index 10 (C = 1)
    @test all(maximum(get_spec(c); dims=1) .≈ 1)
    c2 = Chroma(stft; norm=2)
    @test all(isapprox.(sqrt.(sum(abs2, get_spec(c2); dims=1)), 1))
    cn = Chroma(stft; norm=nothing)
    @test get_spec(cn) ≈ get_data(fb) * get_spec(stft)
    @test_throws DimensionMismatch Chroma(Stft(a440, sr; winsize=1024), fb)
    tz = Tonnetz(c)
    @test size(get_spec(tz)) == (6, get_nframes(stft))
    @test get_parent(tz) === c
    @test Tonnetz(stft) isa Tonnetz
    @test isapprox(hz_to_octs(440 / 16), 0; atol=1e-12)

    audio = Audio911.load(wav_file; format=Float64)
    st = Stft(audio; winsize=1024, winstep=256)
    sc = SpectralContrast(st)
    @test size(get_spec(sc)) == (7, get_nframes(st))
    @test all(isfinite, get_spec(sc))
    @test get_spec(SpectralContrast(st; linear=true)) != get_spec(sc)
    @test_throws ArgumentError SpectralContrast(st; nbands=7)
    pf = PolyFeatures(st; order=1)
    @test size(get_spec(pf)) == (2, get_nframes(st))
    flat = Audio911._derived(st, fill(3.0, size(get_spec(st))), :flat)
    p0 = get_spec(PolyFeatures(flat; order=1))
    @test all(isapprox.(p0[1, :], 0; atol=1e-9)) && all(isapprox.(p0[2, :], 3; atol=1e-6))
    @test size(get_spec(PolyFeatures(st; order=2)), 1) == 3
    # works on a wavelet front end and in Float32
    cw = Cwt(Float32.(a440), sr; winsize=2048, winstep=512, voices=12, freqrange=(100, 4000))
    cc = Chroma(cw)
    @test eltype(get_spec(cc)) == Float32
    @test all(argmax(view(get_spec(cc), :, j)) == 10 for j in 1:get_nframes(cc))
end

@testset "onset, tempo, beats" begin
    csr = 22050
    times = collect(0.5:0.5:7.5)            # 120 BPM click train
    y = clicks(times; sr=csr, click_freq=1000, click_duration=0.05, length=8csr)
    st = Stft(y, csr; winsize=2048, winstep=512, type=hanning, center=true)
    mel = MelSpec(st; nbands=64, win_norm=false)
    env = OnsetStrength(mel)
    @test length(get_data(env)) == get_nframes(mel)
    @test get_data(env)[1] == 0
    @test all(get_data(env) .>= 0)
    idx = onset_detect(env)
    det = get_times(env)[idx]
    @test length(det) == length(times)
    @test all(minimum(abs.(det .- tt)) < 0.04 for tt in times)
    bpm = tempo(env)
    @test isapprox(bpm, 120; rtol=0.05)
    tg = Tempogram(env; win_length=256)
    @test size(get_spec(tg)) == (256, get_nframes(env))
    @test get_freq(tg)[1] == Inf && get_freq(tg)[2] ≈ 60 * csr / 512
    @test all(isapprox.(get_spec(tg)[1, :], 1) .| (get_spec(tg)[1, :] .== 0))
    b, beats = beat_track(env)
    @test isapprox(b, 120; rtol=0.05)
    bt = get_times(env)[beats]
    @test length(bt) ≥ length(times) - 2
    @test mean(minimum(abs.(bt .- tt)) for tt in times[2:end-1]) < 0.05
    @test onset_detect(env; delta=2.0) == Int[]
    @test peak_pick([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]; pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.1, wait=1) == [2, 5]
    @test beat_track(OnsetStrength(MelSpec(Stft(zeros(4csr), csr; winsize=2048, winstep=512); win_norm=false)))[2] == Int[]
    @test get_data(OnsetStrength(mel; lag=2, max_size=3, detrend=true)) isa Vector
    @test_throws ArgumentError OnsetStrength(mel; lag=0)
end

@testset "hpss, gates, pcen" begin
    y = tone220 .+ clicks(collect(0.25:0.5:1.75); sr, click_duration=0.02, length=length(tone220))
    st = Stft(y, sr; winsize=1024, winstep=256, type=hanning)
    h = Hpss(st; kernel=(17, 17))
    H, P = get_harmonic(h), get_percussive(h)
    @test H isa DerivedSpec && P isa DerivedSpec
    @test get_name(H) == :harmonic && get_name(P) == :percussive
    @test get_freq(H) === get_freq(st) && get_spectrum(H) === power
    @test get_frontend(MelSpec(H; nbands=20)) === H
    mh, mp = get_masks(h)
    @test all(isapprox.(mh .+ mp, 1; atol=1e-6))
    k = argmin(abs.(get_freq(st) .- 220))
    @test sum(get_spec(H)[k, :]) > sum(get_spec(P)[k, :])
    @test sum(get_spec(P)) > 0
    hard = Hpss(st; power=Inf)
    @test all(x -> x in (0, 0.5, 1), get_masks(hard)[1])
    @test_throws ArgumentError Hpss(st; margin=0.5)
    @test Mfcc(MelSpec(H; nbands=20); ncoeffs=10) isa Mfcc
    @test length(get_data(SpectralCentroid(P))) == get_nframes(st)

    # time-domain gate: the quiet half is silenced, the loud half is kept
    x = vcat(0.5 .* tone220[1:sr], 0.001 .* tone220[1:sr])
    g = noisegate(x, sr; threshold=-30, attack=0.005, release=0.01, hold=0.01)
    @test maximum(abs, g[sr÷2:sr-1000]) > 0.45
    @test maximum(abs, g[sr+4000:end]) < 1e-4
    @test length(g) == length(x)

    # spectral gate acts only inside the range
    sg = SpectralGate(st; threshold=-20, freqrange=(1000, 8000))
    S = get_spec(st); G = get_spec(sg)
    lo = get_freq(st) .< 1000
    @test G[lo, :] == S[lo, :]
    @test sum(G[.!lo, :]) < sum(S[.!lo, :])
    @test get_name(sg) == :gated
    @test get_data(LinSpec(sg; freqrange=(100, 1000))) isa AbstractMatrix

    pc = pcen(MelSpec(st; nbands=20))
    @test size(get_spec(pc)) == (20, get_nframes(st))
    @test all(isfinite, get_spec(pc))
    @test get_name(pc) == :pcen
    @test eltype(get_spec(pcen(MelSpec(Stft(Float32.(y), sr; winsize=1024); nbands=20)))) == Float32
end
