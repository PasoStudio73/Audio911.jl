using Test
using Audio911
using MAT

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
af_files_dir()      = joinpath(dirname(@__FILE__), "audioflux_files", "fbank")
af_file(filename)   = joinpath(af_files_dir(), filename)

wav_file = test_file("test.wav")

# ---------------------------------------------------------------------------- #
#                         scales and styles, structural                        #
# ---------------------------------------------------------------------------- #
@testset "audioFlux scales and styles" begin
    sr = 16000
    for scale in (linspace, erb, logspace)
        fb = auditory_fbank(sr; nfft=512, nbands=20, scale, freqrange=(100, 6000), norm=none_norm)
        @test size(get_data(fb)) == (20, 257)
        @test length(get_freq(fb)) == 20
        @test issorted(get_freq(fb))
        @test get_scale(fb) == nameof(scale)
        @test all(≥(0), get_data(fb))
    end
    # linspace / logspace put the first and last centre on the range bounds
    @test get_freq(auditory_fbank(sr; nbands=20, scale=linspace, freqrange=(100, 6000)))[[1, end]] ≈ [100, 6000]
    @test get_freq(auditory_fbank(sr; nbands=20, scale=logspace, freqrange=(100, 6000)))[[1, end]] ≈ [100, 6000]
    lg = get_freq(auditory_fbank(sr; nbands=20, scale=logspace, freqrange=(100, 6000)))
    @test all(isapprox.(lg[2:end] ./ lg[1:end-1], lg[2] / lg[1]; rtol=1e-6))
    # erb bounds are the outer edges, like the mel scale
    @test 0 < get_freq(auditory_fbank(sr; nbands=20, scale=erb, freqrange=(100, 7000)))[1]
    # octave: 12 bins per octave from the nearest bin to the low end
    fb = auditory_fbank(sr; nfft=2048, nbands=60, scale=octave, freqrange=(100, 8000), bins_per_octave=12)
    f = get_freq(fb)
    @test isapprox(f[13] / f[1], 2; rtol=1e-6)
    @test isapprox(f[1], 440 * 2.0^(round(12 * log2(100 / 440)) / 12); rtol=1e-6)
    fb24 = auditory_fbank(sr; nfft=2048, nbands=60, scale=octave, freqrange=(100, 8000), bins_per_octave=24)
    @test isapprox(get_freq(fb24)[25] / get_freq(fb24)[1], 2; rtol=1e-6)
    @test_throws ArgumentError auditory_fbank(sr; nbands=20, scale=octave, freqrange=(0, 8000))
    @test_throws ArgumentError auditory_fbank(sr; nbands=20, scale=logspace, freqrange=(0, 8000))
    @test_throws ArgumentError auditory_fbank(sr; nbands=20, scale=sum)
    @test_throws ArgumentError auditory_fbank(sr; nbands=200, scale=octave, freqrange=(100, 8000))  # beyond the grid

    # window styles
    for style in (etsi, point, rect, hanning, hamming, blackman, bohman, kaiser, gauss)
        fb = auditory_fbank(sr; nfft=512, nbands=26, style, norm=none_norm)
        W = get_data(fb)
        @test size(W) == (26, 257)
        @test all(0 .≤ W .≤ 1 + 1e-12)
        @test all(maximum(W, dims=2) .≈ 1)           # every band peaks at 1
    end
    @test sum(get_data(auditory_fbank(sr; nbands=26, style=point, norm=none_norm))) == 26
    @test_throws ArgumentError auditory_fbank(sr; nbands=26, style=hanning, domain=:warped)
    @test_throws ArgumentError auditory_fbank(sr; nbands=26, style=sum)

    # windows
    @test bohman(1) == [1.0]
    @test bohman(9)[5] ≈ 1
    @test bohman(9)[1] ≈ 0 atol=1e-12
    @test kaiser(9)[5] ≈ 1
    @test gauss(9)[5] ≈ 1
    @test gauss(9)[1] ≈ exp(-0.5 * 2.5^2)
    @test triangular(9) == bartlett(9)
    @test point(9)[5] == 1 && sum(point(9)) == 1
    fr = Frames(zeros(4000), 8000; winsize=256, winstep=128, type=bohman)
    @test length(get_window(fr)) == 256

    # the stages accept every scale and style
    audio = Audio911.load(wav_file; format=Float64)
    stft  = Stft(audio; winsize=512, winstep=256)
    for scale in (htk, slaney, linspace, erb, logspace)
        @test MelSpec(stft; nbands=20, scale, freqrange=(100, 6000)) isa MelSpec
    end
    @test MelSpec(stft; nbands=48, scale=octave, freqrange=(100, 7000), bins_per_octave=12) isa MelSpec
    @test MelSpec(stft; nbands=20, style=gauss) isa MelSpec
    @test_throws ArgumentError MelSpec(stft; nbands=20, scale=bark)
    # a chromagram folds the octave scale
    ch = Chroma(MelSpec(stft; nbands=48, scale=octave, freqrange=(100, 7000), norm=none_norm))
    @test size(get_spec(ch)) == (12, get_nframes(stft))

    # Float32
    a32 = Audio911.load(wav_file; format=Float32)
    s32 = Stft(a32; winsize=512, winstep=256)
    @test eltype(get_data(MelSpec(s32; nbands=20, scale=erb, style=hanning))) == Float32
end

# ---------------------------------------------------------------------------- #
#                              against audioFlux                               #
# ---------------------------------------------------------------------------- #
# obj = af.BFT(num=num, radix2_exp=9, samplate=sr, low_fre=low, high_fre=high,
#              bin_per_octave=bpo, window_type=WindowType.HANN, slide_length=256,
#              scale_type=..., style_type=..., normal_type=..., data_type=POWER)
# spec = obj.bft(x, result_type=1)
@testset "filterbank spectrograms against audioFlux" begin
    audio  = Audio911.load(wav_file; format=Float64)
    frames = Frames(audio; winsize=512, winstep=256, type=hanning)
    stft   = Stft(frames)

    # the linear BFT is the STFT power spectrogram
    mat = MAT.matread(af_file("linear.mat"))
    @test Int(mat["time_len"]) == length(frames)
    @test isapprox(get_spec(stft), mat["spec"]; rtol=1e-4, atol=1e-6 * maximum(mat["spec"]))

    cases = [
        ("mel_slaney_none",      htk,      triangular, none_norm),
        ("mel_slaney_area",      htk,      triangular, area),
        ("mel_slaney_bw",        htk,      triangular, bandwidth),
        ("bark_slaney_none",     bark,     triangular, none_norm),
        ("erb_slaney_none",      erb,      triangular, none_norm),
        ("linspace_slaney_none", linspace, triangular, none_norm),
        ("octave_slaney_none",   octave,   triangular, none_norm),
        ("log_slaney_none",      logspace, triangular, none_norm),
        ("mel_hann_none",        htk,      hanning,    none_norm),
        ("mel_hamm_area",        htk,      hamming,    area),
        ("mel_blackman_none",    htk,      blackman,   none_norm),
        ("mel_bohman_none",      htk,      bohman,     none_norm),
        ("mel_kaiser_none",      htk,      kaiser,     none_norm),
        ("mel_gauss_none",       htk,      gauss,      none_norm),
        ("mel_rect_none",        htk,      rect,       none_norm),
        ("mel_point_none",       htk,      point,      none_norm),
    ]
    for (name, scale, style, norm) in cases
        mat = MAT.matread(af_file(name * ".mat"))
        ref = mat["spec"]
        num = Int(mat["num"]); low = Float64(mat["low"]); high = Float64(mat["high"])
        freqrange = (round(Int, low), round(Int, high))
        fb = auditory_fbank(stft; nbands=num, scale, style, norm, freqrange, bins_per_octave=Int(mat["bpo"]))
        @test vec(get_freq(fb)) ≈ vec(mat["freq"]) rtol=2e-4
        spec = scale === bark ? BarkSpec(stft, fb) : MelSpec(stft, fb)
        S = get_spec(spec)
        @test size(S) == size(ref)
        @test isapprox(S, ref; rtol=1e-3, atol=1e-5 * maximum(ref))
    end

    # audioFlux's ETSI style is a bin-based triangle (not the ETSI standard's etsi_fbank)
    mat = MAT.matread(af_file("mel_etsi_none.mat"))
    fb  = auditory_fbank(stft; nbands=26, style=etsi, norm=none_norm)
    @test isapprox(get_spec(MelSpec(stft, fb)), mat["spec"]; rtol=1e-3, atol=1e-5 * maximum(mat["spec"]))
    fb2 = etsi_fbank(16000; nfft=512, nbands=26, freqrange=(0, 8000))
    @test !isapprox(get_spec(MelSpec(stft, fb2)), mat["spec"]; rtol=1e-3)
end
