using Test
using Audio911
using MAT

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
af_files_dir()      = joinpath(dirname(@__FILE__), "audioflux_files", "hpss")
af_file(filename)   = joinpath(af_files_dir(), filename)

wav_file = test_file("test.wav")
rel_l2(a, b) = sqrt(sum(abs2, a .- b)) / sqrt(sum(abs2, b))

@testset "Hpss signals" begin
    sr = 16000
    t  = (0:2sr-1) ./ sr
    tone = 0.5 .* sin.(2π * 440 .* t)
    clk  = zeros(length(t)); clk[4000:4000:end] .= 1
    x = tone .+ clk
    stft = Stft(Frames(x, sr; winsize=1024, winstep=256, type=hanning, center=true); keep_complex=true)
    h = Hpss(stft; kernel=(17, 17), edge=:zero)
    yh, yp = get_harmonic_signal(h), get_percussive_signal(h)
    @test length(yh) == length(yp) == length(x)
    @test eltype(yh) == Float64
    # the masks sum to one, so the two signals add up to the input
    @test isapprox(yh .+ yp, x; rtol=1e-8, atol=1e-8)
    # the tone goes to the harmonic part and the clicks to the percussive one
    mid = 3000:length(x)-3000
    @test rel_l2(yh[mid], tone[mid]) < 0.2
    @test sum(abs2, yp[mid] .- clk[mid]) < sum(abs2, clk[mid])
    @test length(get_harmonic_signal(h; length=100)) == 100
    @test length(get_percussive_signal(h; method=:ola)) == length(x)
    # zero-padded edges only change the borders
    hs = Hpss(stft; kernel=(17, 17))
    Mz, Ms = get_masks(h)[1], get_masks(hs)[1]
    @test Mz[20:end-20, 20:end-20] == Ms[20:end-20, 20:end-20]
    @test Mz != Ms
    @test_throws ArgumentError Hpss(stft; edge=:reflect)
    @test_throws ArgumentError get_harmonic_signal(Hpss(MelSpec(stft)))
    # Float32 stays Float32
    s32 = Stft(Frames(Float32.(x), sr; winsize=1024, winstep=256, center=true))
    @test eltype(get_percussive_signal(Hpss(s32))) == Float32
end

@testset "Hpss against audioFlux" begin
    audio = Audio911.load(wav_file; format=Float64)
    mat = MAT.matread(af_file("hpss.mat"))
    n = Int(mat["n"])
    # obj = af.HPSS(radix2_exp=11, window_type=WindowType.HAMM, slide_length=512,
    #               h_order=21, p_order=31); h, p = obj.hpss(x)
    # (audioFlux forces the hop to fft/4 whatever slide_length says)
    for (win, hop, type, kernel, hk, pk) in ((2048, 512, hamming, (21, 31), "h", "p"),
                                             (1024, 256, hanning, (11, 5), "h2", "p2"))
        # obj2 = af.HPSS(radix2_exp=10, window_type=WindowType.HANN, slide_length=100,
        #                h_order=11, p_order=5); h2, p2 = obj2.hpss(x)
        stft = Stft(Frames(audio; winsize=win, winstep=hop, type))
        h = Hpss(stft; kernel, edge=:zero)
        ref_h, ref_p = vec(mat[hk]), vec(mat[pk])
        L = length(ref_h)
        yh = get_harmonic_signal(h; length=L)
        yp = get_percussive_signal(h; length=L)
        @test rel_l2(yh, ref_h) < 1e-4
        @test rel_l2(yp, ref_p) < 1e-4
        # audioFlux returns (frames - 1) · hop + fft samples (no incomplete tail frame)
        @test L == (get_nframes(stft) - 1) * hop + win ≤ n
    end
end
