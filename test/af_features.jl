using Test
using Audio911
using MAT

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
af_file(filename)   = joinpath(dirname(@__FILE__), "audioflux_files", "features", filename)
wav_file = test_file("test.wav")
rel_l2(a, b) = sqrt(sum(abs2, vec(a) .- vec(b))) / max(sqrt(sum(abs2, b)), 1e-30)

@testset "audioFlux features, structural" begin
    audio  = Audio911.load(wav_file; format=Float64)
    frames = Frames(audio; winsize=512, winstep=256, type=hanning)
    stft   = Stft(frames; spectrum=magnitude)
    n = length(frames)

    c, d1, d2 = xxcc_standard(stft)
    @test c isa Mfcc && d1 isa Delta && d2 isa Delta
    @test size(get_spec(c)) == (13, n) && size(get_spec(d2)) == (13, n)
    @test get_spec(c)[1, :] ≈ log.(max.(vec(sum(get_spec(stft), dims=1)), 1e-8))
    cp, _, _ = xxcc_standard(stft; energy_mode=:prepend)
    @test size(get_spec(cp)) == (14, n)
    @test get_spec(cp)[2:end, :] ≈ get_spec(Mfcc(stft; ncoeffs=13, rect=mlog, floor=1e-8))
    ca, _, _ = xxcc_standard(stft; energy=nothing)
    @test get_spec(ca) ≈ get_spec(Mfcc(stft; ncoeffs=13, rect=mlog, floor=1e-8))
    @test size(get_spec(xxcc_standard(MelSpec(Stft(frames); nbands=40); ncoeffs=20)[1])) == (20, n)
    @test size(get_spec(Mfcc(Stft(frames); energy=raw_energy, energy_mode=:append))) == (129, n)

    d = Deconv(stft)
    @test size(get_timbre(d)) == size(get_pitch(d)) == size(get_spec(stft))
    @test get_data(d) === get_timbre(d)
    @test get_times(d) == get_times(stft)
    # the timbre part is the autocorrelation of the frame's spectrum: its peak is at lag 0
    @test all(argmax(get_timbre(d)[:, j]) == 1 for j in 1:n if sum(get_spec(stft)[:, j]) > 0)

    cg = Cepstrogram(Frames(audio; winsize=512, winstep=256, type=rect); ncep=12)
    @test size(get_data(cg)) == size(get_envelope(cg)) == size(get_details(cg)) == (257, n)
    @test get_quefrency(cg)[2] ≈ 1 / 16000
    @test get_freq(cg)[end] ≈ 8000
    # envelope and details add up to the log power spectrum, up to the one
    # quefrency audioFlux counts in both
    P = log.(max.(get_spec(Stft(Frames(audio; winsize=512, winstep=256, type=rect))), 1e-16))
    y = get_data(cg)
    @test rel_l2(get_envelope(cg) .+ get_details(cg), P) < 0.05
    @test_throws ArgumentError Cepstrogram(frames; ncep=0)

    # a pure tone: few zero crossings per frame, a large energy-to-crossing ratio
    sr = 16000; t = (0:sr-1) ./ sr
    tone = Frames(sin.(2π * 100 .* t), sr; winsize=512, winstep=256)
    noise = Frames(randn(sr) .* 0.5, sr; winsize=512, winstep=256)
    @test sum(get_data(Ezr(tone))) > 10 * sum(get_data(Ezr(noise)))
    @test all(abs.(get_data(Zcr(tone; strict=true, rate=false)) .- get_data(Zcr(tone; rate=false))) .≤ 1)
    # the audioFlux harmonic ratio is high for a tone and low for noise
    ht = get_data(HarmonicRatio(Frames(sin.(2π * 220 .* t), sr; winsize=1024, winstep=512, type=hamming); method=:audioflux))
    hn = get_data(HarmonicRatio(Frames(randn(sr), sr; winsize=1024, winstep=512, type=hamming); method=:audioflux))
    @test minimum(ht) > 0.9 && maximum(hn) < 0.6
    @test_throws ArgumentError HarmonicRatio(frames; method=:foo)

    a32 = Audio911.load(wav_file; format=Float32)
    f32 = Frames(a32; winsize=512, winstep=256)
    s32 = Stft(f32; spectrum=magnitude)
    @test eltype(get_timbre(Deconv(s32))) == Float32
    @test eltype(get_data(Cepstrogram(f32))) == Float32
    @test eltype(get_data(Ezr(f32))) == Float32
    @test eltype(get_spec(xxcc_standard(s32)[3])) == Float32
    @test eltype(get_data(HarmonicRatio(f32; method=:audioflux))) == Float32
end

# see test/audioflux_sources/features.py for every call
@testset "audioFlux features against audioFlux" begin
    audio = Audio911.load(wav_file; format=Float64)
    m  = MAT.matread(af_file("features.mat"))
    fr = Frames(audio; winsize=512, winstep=256, type=hanning)
    mag = Stft(fr; spectrum=magnitude)

    # af.XXCC(num=257).xxcc(mag, cc_num=13, rectify_type=LOG | CUBIC_ROOT)
    @test rel_l2(get_spec(Mfcc(mag; ncoeffs=13, rect=mlog, floor=1e-8)), m["xxcc_log"]) < 1e-3
    @test rel_l2(get_spec(Mfcc(mag; ncoeffs=13, rect=cubic_root)), m["xxcc_cbrt"]) < 1e-3
    # xxcc_standard(mag, energy=Σ power, cc_num=13, delta_window_length=9, REPLACE | APPEND)
    for (mode, key) in ((:replace, "std_replace"), (:prepend, "std_append"))
        c, d1, d2 = xxcc_standard(mag; energy=vec(m["energy"]), energy_mode=mode)
        @test rel_l2(get_spec(c), m[key]) < 1e-3
        @test rel_l2(get_spec(d1), m[key * "_d1"]) < 1e-3
        @test rel_l2(get_spec(d2), m[key * "_d2"]) < 1e-3
    end
    # af.Deconv(num=257).deconv(mag)
    d = Deconv(mag)
    @test rel_l2(get_timbre(d), m["deconv_timbre"]) < 1e-5
    @test rel_l2(get_pitch(d), m["deconv_pitch"]) < 1e-5
    # af.Cepstrogram(radix2_exp=9, samplate=sr, window_type=RECT, slide_length=256).cepstrogram(x, cep_num=8)
    cg = Cepstrogram(Frames(audio; winsize=512, winstep=256, type=rect); ncep=8)
    @test rel_l2(get_data(cg), m["cepstrum"]) < 1e-4
    @test rel_l2(get_envelope(cg), m["envelope"]) < 1e-4
    @test rel_l2(get_details(cg), m["details"]) < 1e-3
    # af.Temporal(frame_length=512, slide_length=256, window_type=HANN): energy, rms, zcr, ezr(10)
    @test rel_l2(get_data(Energy(fr; windowed=true)), m["t_energy"]) < 1e-5
    @test rel_l2(get_data(Rms(fr; windowed=true)), m["t_rms"]) < 1e-5
    @test rel_l2(get_data(Zcr(fr; windowed=true, strict=true)), m["t_zcr"]) < 1e-5
    @test rel_l2(get_data(Ezr(fr; gamma=10)), m["t_ezr"]) < 1e-5
    # af.HarmonicRatio(samplate=sr, low_fre=32.703, radix2_exp=12, window_type=HAMM, slide_length=1024)
    hr = HarmonicRatio(Frames(audio; winsize=4096, winstep=1024, type=hamming); method=:audioflux, fmin=32.703)
    @test rel_l2(get_data(hr), m["hr"]) < 1e-4
    hr = HarmonicRatio(Frames(audio; winsize=1024, winstep=256, type=hamming); method=:audioflux, range=(50, 400))
    @test rel_l2(get_data(hr), m["hr_small"]) < 1e-4
    # af.chroma_linear(x, chroma_num=12, radix2_exp=9, samplate=sr, window_type=HANN, slide_length=256)
    @test rel_l2(get_spec(Chroma(Stft(fr))), m["chroma_linear"]) < 1e-4
end
