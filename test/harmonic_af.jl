using Test
using Audio911
using MAT

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
af_files_dir()      = joinpath(dirname(@__FILE__), "audioflux_files", "harmonic")
af_file(filename)   = joinpath(af_files_dir(), filename)

wav_file = test_file("test.wav")

@testset "harmonic_count" begin
    sr = 32000
    t = (0:sr-1) ./ sr
    for nh in (1, 4, 8)
        y = 0.2 .* sum(sin.(2π * 220k .* t) ./ k for k in 1:nh)
        c = harmonic_count(Stft(Frames(y, sr; winsize=4096, winstep=1024, type=hamming)))
        @test all(==(nh), c)
    end
    y = 0.2 .* sum(sin.(2π * 220k .* t) ./ k for k in 1:8)
    s = Stft(Frames(y, sr; winsize=4096, winstep=1024, type=hamming))
    @test all(==(4), harmonic_count(s; count_range=(100, 1000)))     # 220 … 880 Hz
    @test harmonic_count(Stft(Frames(y, sr; winsize=4096, winstep=1024, type=hamming); spectrum=magnitude)) ==
          harmonic_count(s)
    @test all(==(0), harmonic_count(Stft(Frames(zeros(sr), sr; winsize=4096, winstep=1024))))
    @test length(harmonic_count(s)) == get_nframes(s)
    @test_throws ArgumentError harmonic_count(s; range=(100, 100))
end

@testset "harmonic_count against audioFlux" begin
    mat = MAT.matread(af_file("harmonic.mat"))
    audio = Audio911.load(wav_file; format=Float64)
    high = Int(mat["high"])
    s = Stft(Frames(audio; winsize=4096, winstep=1024, type=hamming))
    # obj = af.Harmonic(radix2_exp=12, samplate=sr, slide_length=1024, window_type=WindowType.HAMM,
    #                   low_fre=27, high_fre=high); obj.harmonic_count(x, 27, high)
    c = harmonic_count(s; range=(27, high))
    ref = Int.(vec(mat["wav_full"]))
    @test length(c) == length(ref)
    @test c == ref
    # obj.harmonic_count(x, 100, 1000)
    cb = harmonic_count(s; range=(27, high), count_range=(100, 1000))
    refb = Int.(vec(mat["wav_band"]))
    @test cb == refb
    # the synthetic tone with 3, 6 and 10 partials (float32 samples)
    y = Float64.(vec(mat["tone"]))
    st = Stft(Frames(y, 32000; winsize=4096, winstep=1024, type=hamming))
    ct = harmonic_count(st; range=(27, 4000))
    reft = Int.(vec(mat["tone_count"]))
    @test ct == reft
end
