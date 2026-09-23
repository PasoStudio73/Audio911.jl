using Test
using Audio911
const rfft = Audio911.FFTW.rfft
using MAT

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
af_files_dir()      = joinpath(dirname(@__FILE__), "audioflux_files", "stretch")
af_file(filename)   = joinpath(af_files_dir(), filename)

wav_file = test_file("test.wav")
rel_l2(a, b) = sqrt(sum(abs2, a .- b)) / sqrt(sum(abs2, b))
# the dominant frequency of a signal (away from its ends)
peak_hz(y, sr) = (Y = abs.(rfft(y[4097:end-4096] .* Audio911.hanning(length(y) - 8192))); (argmax(Y) - 1) * sr / (length(y) - 8192))

@testset "phase vocoder, time stretch, pitch shift" begin
    sr = 16000
    t = (0:2sr-1) ./ sr
    x = 0.5 .* sin.(2π * 440 .* t)
    C = get_complex(Stft(Frames(x, sr; winsize=1024, winstep=256)))
    D = phase_vocoder(C, 1; hop=256)
    @test size(D) == size(C)
    @test abs.(D) ≈ abs.(C)
    @test size(phase_vocoder(C, 2; hop=256), 2) == cld(size(C, 2), 2)
    @test size(phase_vocoder(C, 0.5; hop=256), 2) == 2size(C, 2)
    @test_throws ArgumentError phase_vocoder(C, 0; hop=256)

    for rate in (0.5, 0.8, 1.25, 2.0)
        y = time_stretch(x, rate)
        @test length(y) == round(Int, length(x) / rate)
        @test isapprox(peak_hz(y, sr), 440; atol=2)
    end
    @test eltype(time_stretch(Float32.(x), 1.5)) == Float32
    for n in (-7, -2, 4, 12)
        y = pitch_shift(x, n)
        @test length(y) == length(x)
        @test isapprox(peak_hz(y, sr), 440 * 2^(n / 12); rtol=0.01)
    end
    a = AudioFile(hcat(x, x), sr; mono=false)
    b = pitch_shift(a, 3)
    @test get_sr(b) == sr && size(get_data(b)) == (length(x), 2)
    @test get_data(b)[:, 1] ≈ pitch_shift(x, 3)
    @test size(get_data(time_stretch(a, 2))) == (length(x) ÷ 2, 2)

    # band-limited resampling keeps a tone and the element type
    y = Audio911.resample(x, sr, 8000; method=:sinc)
    @test length(y) == length(x) ÷ 2
    @test isapprox(peak_hz(y, 8000), 440; atol=2)
    y32 = Audio911.resample(Float32.(x), sr, 22050; method=:sinc, quality=:fast)
    @test eltype(y32) == Float32 && length(y32) == floor(Int, length(x) * 22050 / sr)
    @test size(Audio911.resample(hcat(x, x), sr, 8000; method=:sinc)) == (length(x) ÷ 2, 2)
    @test Audio911.resample(x, sr, 8000; method=:sinc, scale=true) ≈ y ./ sqrt(0.5)
    @test_throws ArgumentError Audio911.resample(x, sr, 8000; method=:foo)
    @test_throws ArgumentError Audio911.resample(x, sr, 8000; quality=:best)
    @test_throws ArgumentError Audio911.resample(x, sr, 8000; method=:sinc, quality=:ultra)
end

@testset "stretch and resampling against audioFlux" begin
    mat = MAT.matread(af_file("stretch.mat"))
    audio = Audio911.load(wav_file; format=Float64)
    x = vec(get_data(audio)); sr = get_sr(audio)
    # obj = af.TimeStretch(radix2_exp=12, slide_length=1024, window_type=WindowType.HANN)
    # y = obj.time_stretch(x, rate): the C call returns round(n / rate) samples; the
    # wrapper's buffer continues with unnormalised overlap-add values
    for (name, rate) in (("0_5", 0.5), ("0_8", 0.8), ("1_5", 1.5))
        ref = vec(mat["ts_" * name])
        y = time_stretch(x, rate)
        @test rel_l2(y, ref[1:length(y)]) < 2e-3      # float32 phase accumulation
    end
    # obj = af.PitchShift(radix2_exp=12, slide_length=1024, window_type=WindowType.HANN)
    # y = obj.pitch_shift(x, n_semitone, samplate=sr)
    for n in (-5, 3, 7)
        ref = vec(mat["ps_" * replace(string(n), "-" => "m")])
        y = pitch_shift(x, n)
        @test length(y) == length(ref)
        @test rel_l2(y, ref) < 1e-3
    end
    # obj = af.Resample(qual_type, is_scale); obj.set_samplate(sr, target); y = obj.resample(x)
    for q in (:best, :mid, :fast), (target, scale) in ((sr ÷ 2, false), (round(Int, 1.5sr), true), (11025, false))
        ref = vec(mat["rs_$(q)_$(target)"])
        y = Audio911.resample(x, sr, target; method=:sinc, quality=q, scale)
        @test length(y) == length(ref)
        @test rel_l2(y, ref) < 1e-4
    end
    # obj = af.WindowResample(zero_num=32, nbit=9, win_type=WindowType.HANN, value=0, roll_off=0.9)
    ref = vec(mat["wrs_hann_12000"])
    y = Audio911.resample(x, sr, 12000; method=:sinc, nzeros=32, window=hanning, rolloff=0.9)
    @test length(y) == length(ref)
    @test rel_l2(y, ref) < 1e-4
end
