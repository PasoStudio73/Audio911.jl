using Test
using Audio911
using MAT

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
af_files_dir()      = joinpath(dirname(@__FILE__), "audioflux_files", "cqt")
af_file(filename)   = joinpath(af_files_dir(), filename)

wav_file = test_file("test.wav")

# ---------------------------------------------------------------------------- #
#                                 structural                                   #
# ---------------------------------------------------------------------------- #
@testset "Cqt front end" begin
    audio  = Audio911.load(wav_file; format=Float64)
    frames = Frames(audio; winsize=512, winstep=256, type=hanning)
    cqt    = Cqt(frames; nbins=84)

    @test cqt isa Cqt{Float64}
    @test get_spec(cqt) === get_data(cqt)
    @test size(get_spec(cqt)) == (84, length(frames))
    @test length(get_freq(cqt)) == 84
    @test issorted(get_freq(cqt))
    @test get_freq(cqt)[1] ≈ 32.703
    @test get_freq(cqt)[13] ≈ 2 * 32.703
    @test get_sr(cqt) == 16000
    @test get_spectrum(cqt) === power
    @test all(>=(0), get_spec(cqt))
    @test get_frames(cqt) === frames && get_parent(cqt) === frames
    @test get_step(cqt) == 256 && get_winsize(cqt) == 512
    @test get_times(cqt) == get_times(Stft(frames))
    @test get_window(cqt) === nothing && get_winnorm(cqt) == 1
    @test ispow2(get_nfft(cqt))
    @test get_nfft(cqt) ≥ get_bandwidth(cqt)[1] > get_bandwidth(cqt)[end]
    @test get_nfft(cqt) == 64 * nextpow(2, ceil(Int, get_bandwidth(cqt)[73]))
    @test get_frontend(MelSpec(cqt; nbands=20, freqrange=(50, 3900))) === cqt

    # the complex coefficients are recomputed on request or kept
    C = get_complex(cqt)
    @test size(C) == size(get_spec(cqt))
    @test abs2.(C) ≈ get_spec(cqt)
    kept = Cqt(frames; nbins=84, keep_complex=true)
    @test get_complex(kept) === kept.cplx
    @test get_complex(kept) ≈ C
    @test get_phase(kept) ≈ angle.(C)
    @test get_spec(Cqt(frames; nbins=84, spectrum=magnitude)) ≈ abs.(C)

    # the downstream chain runs unchanged on a constant-Q front end
    mel  = MelSpec(cqt; nbands=26, freqrange=(100, 3900))
    @test size(get_data(mel)) == (length(frames), 26)
    mfcc = Mfcc(mel; ncoeffs=13)
    @test size(get_data(mfcc)) == (length(frames), 13)
    @test size(get_data(Delta(mfcc))) == size(get_data(mfcc))
    # audioFlux's cqcc: the cepstrum of the constant-Q spectrogram itself
    cqcc = Mfcc(cqt; ncoeffs=20, rect=mlog, floor=1e-8)
    @test size(get_data(cqcc)) == (length(frames), 20)
    @test get_spec(cqcc) ≈ (dct_ortho(Float64, 84) * mlog.(max.(get_spec(cqt), 1e-8)))[1:20, :]
    lin = LinSpec(cqt; freqrange=(100, 4000))
    @test all(100 .<= get_freq(lin) .<= 4000)
    for D in (SpectralCentroid, SpectralCrest, SpectralDecrease, SpectralEntropy,
              SpectralFlatness, SpectralFlux, SpectralKurtosis, SpectralRolloff,
              SpectralSkewness, SpectralSlope, SpectralSpread, SpectralBandwidth)
        x = D(cqt)
        @test length(get_data(x)) == length(frames)
        @test all(isfinite, get_data(x))
    end
    ch = Chroma(cqt)
    @test size(get_spec(ch)) == (12, length(frames))
    @test size(get_spec(Tonnetz(ch))) == (6, length(frames))
    @test size(get_spec(SpectralContrast(cqt; nbands=4, fmin=100))) == (5, length(frames))
    @test_throws ArgumentError MelSpec(cqt; nbands=20)          # default range beyond the CQT grid
    @test length(get_data(OnsetStrength(cqt))) == length(frames)
    @test size(get_spec(get_harmonic(Hpss(cqt)))) == size(get_spec(cqt))
    @test size(get_spec(pcen(cqt))) == size(get_spec(cqt))

    # a pure tone peaks at its bin
    sr = 16000
    t  = (0:2sr-1) ./ sr
    for f0 in (220.0, 440.0, 1000.0)
        c = Cqt(sin.(2π * f0 .* t), sr; winsize=1024, winstep=512, center=true, nbins=84)
        prof = vec(sum(get_spec(c), dims=2))
        peak = get_freq(c)[argmax(prof)]
        @test isapprox(peak, f0; rtol=2^(1/24) - 1 + 1e-6)
    end

    # variable Q shortens the low kernels
    vqt = Cqt(frames; gamma=20)
    @test all(get_bandwidth(vqt) .< get_bandwidth(cqt))
    @test get_bandwidth(vqt)[end] / get_bandwidth(cqt)[end] > get_bandwidth(vqt)[1] / get_bandwidth(cqt)[1]

    # options
    @test Cqt(frames; norm=none_norm) isa Cqt
    @test Cqt(frames; norm=bandwidth) isa Cqt
    @test Cqt(frames; window=hamming, factor=0.5) isa Cqt
    @test get_nfft(Cqt(frames; factor=0.5)) ≤ get_nfft(cqt)
    @test Cqt(audio; winsize=1024, winstep=256, fmin=65.406, bins_per_octave=24, nbins=120) isa Cqt
    @test_throws ArgumentError Cqt(frames; fmin=0)
    @test_throws ArgumentError Cqt(frames; nbins=200)          # reaches Nyquist
    @test_throws ArgumentError Cqt(frames; norm=area, spectrum=sum)
    @test_throws ArgumentError Cqt(frames; gamma=-1)

    # Float32 stays Float32
    a32 = Audio911.load(wav_file; format=Float32)
    c32 = Cqt(a32; winsize=512, winstep=256, nbins=60)
    @test c32 isa Cqt{Float32}
    @test eltype(get_spec(c32)) == Float32
    @test eltype(get_freq(c32)) == Float32
    @test eltype(get_complex(c32)) == ComplexF32
    @test eltype(get_data(Mfcc(MelSpec(c32; nbands=20, freqrange=(50, 900))))) == Float32
end

# ---------------------------------------------------------------------------- #
#                            against audioFlux                                 #
# ---------------------------------------------------------------------------- #
# audioFlux centres frame i on sample i·slide and drops the signal tail beyond
# the last full hop before padding; the same grid is `Frames(...; winstep=slide,
# center=true)` on a tail-zeroed signal (winsize only fixes the grid). Its top
# octave is computed at the full rate exactly as here; the lower octaves run on
# a 2:1 decimated copy (fast resampler, float32) with the same kernel sampling
# as the per-octave FFT lengths here, so they agree within the resampler's
# error (a few percent). The VQT reuses the top octave's kernel lengths for the decimated
# octaves, so only its top octave is a variable-Q reference.
row_err(a, b) = sqrt(sum(abs2, a .- b)) / sqrt(sum(abs2, b))   # relative L2 error of a row

@testset "Cqt against audioFlux" begin
    audio = Audio911.load(wav_file; format=Float64)
    x = vec(get_data(audio))
    n = length(x)
    xz = copy(x); xz[(n ÷ 512) * 512 + 1:end] .= 0

    # obj = af.CQT(num=84, samplate=sr, low_fre=32.703, bin_per_octave=12,
    #              factor=1.0, beta=0.0, thresh=0.01, window_type=WindowType.HANN,
    #              slide_length=512, normal_type=SpectralFilterBankNormalType.AREA,
    #              is_scale=True); spec = obj.cqt(x)
    mat = MAT.matread(af_file("cqt_area.mat"))
    ref = abs2.(mat["spec"])                       # 84 × time
    frames = Frames(xz, 16000; winsize=512, winstep=512, center=true)
    @test length(frames) == Int(mat["time_len"])
    cqt = Cqt(frames; nbins=84, norm=area)
    @test Int(mat["fft_length"]) == get_nfft(cqt) ÷ 64       # audioFlux reports the top-octave FFT
    @test vec(get_freq(cqt)) ≈ vec(mat["freq"]) rtol=1e-5
    S = get_spec(cqt)
    @test isapprox(S, ref; rtol=0.1, atol=1e-4 * maximum(ref))
    cqt_rows_ok(S, ref) = begin
        @test all(row_err(S[k, :], ref[k, :]) < 1e-3 for k in 73:84)   # top octave: same computation
        @test all(row_err(S[k, :], ref[k, :]) < 0.08 for k in 1:48)    # decimated in audioFlux
        @test all(row_err(S[k, :], ref[k, :]) < 0.15 for k in 49:72)   # little energy in this file
    end
    cqt_rows_ok(S, ref)

    # norm NONE
    mat = MAT.matread(af_file("cqt_none.mat"))
    ref = abs2.(mat["spec"])
    S = get_spec(Cqt(frames; nbins=84, norm=none_norm))
    @test isapprox(S, ref; rtol=0.1, atol=1e-4 * maximum(ref))
    cqt_rows_ok(S, ref)

    # white noise (seeded, saved in the fixture): every octave carries energy
    mat = MAT.matread(af_file("cqt_noise.mat"))
    ref = abs2.(mat["spec"])
    xn  = Float64.(vec(mat["x"])); xn[(n ÷ 512) * 512 + 1:end] .= 0
    fn  = Frames(xn, 16000; winsize=512, winstep=512, center=true)
    S = get_spec(Cqt(fn; nbins=84, norm=area))
    @test all(row_err(S[k, :], ref[k, :]) < 1e-3 for k in 73:84)
    @test all(row_err(S[k, :], ref[k, :]) < 0.08 for k in 1:72)

    # VQT on noise: num=144, low_fre=65.406, bin_per_octave=24, beta=20
    mat = MAT.matread(af_file("vqt_noise.mat"))
    ref = abs2.(mat["spec"])
    vqt = Cqt(fn; fmin=65.406, bins_per_octave=24, nbins=144, gamma=20, norm=area)
    @test vec(get_freq(vqt)) ≈ vec(mat["freq"]) rtol=1e-5
    S = get_spec(vqt)
    @test all(row_err(S[k, :], ref[k, :]) < 1e-3 for k in 121:144)
end

@testset "Cqt chroma" begin
    audio = Audio911.load(wav_file; format=Float64)
    x = vec(get_data(audio)); n = length(x)
    xz = copy(x); xz[(n ÷ 512) * 512 + 1:end] .= 0
    frames = Frames(xz, 16000; winsize=512, winstep=512, center=true)
    cqt = Cqt(frames; nbins=84, norm=area)

    fb = cqt_chroma_fbank(cqt)
    W = get_data(fb)
    @test size(W) == (12, 84)
    @test all(sum(W, dims=1) .== 1)                 # every bin in exactly one class
    @test all(sum(W, dims=2) .== 7)                 # 7 octaves per class
    @test W[1, 1] == 1 && W[1, 13] == 1 && W[10, 10] == 1   # C1, C2, A1
    # 24 bins per octave: two bins per class, centred on the semitone
    f24 = [32.703 * 2^((k - 1) / 24) for k in 1:48]
    W24 = get_data(cqt_chroma_fbank(f24, 24))
    @test all(sum(W24, dims=1) .== 1) && all(sum(W24, dims=2) .== 4)
    # the class of a bin does not depend on fmin (row 1 is always C)
    fa = [55.0 * 2^((k - 1) / 12) for k in 1:24]            # starts on A1
    Wa = get_data(cqt_chroma_fbank(fa, 12))
    @test Wa[10, 1] == 1 && Wa[1, 4] == 1                  # A1 → A, C2 → C
    @test_throws ArgumentError cqt_chroma_fbank(f24, 24; nchroma=5)

    ch = Chroma(cqt)
    @test get_fbank(ch) isa ChromaFBank
    @test all(maximum(get_spec(ch), dims=1) .≈ 1)
    @test size(get_spec(Chroma(cqt; norm=nothing))) == (12, length(frames))

    # obj = af.CQT(num=84, samplate=sr, low_fre=32.703, bin_per_octave=12, slide_length=512,
    #              normal_type=AREA); spec = obj.cqt(x)
    # chroma = obj.chroma(spec, chroma_num=12, data_type=POWER, norm_type=MAX)
    mat = MAT.matread(af_file("cqt_chroma.mat"))
    ref = mat["chroma"]
    @test size(ref) == size(get_spec(ch))
    # exact on audioFlux's own CQT, within the decimation error on ours
    refcqt = abs2.(mat["spec"])
    own = get_data(fb) * refcqt
    own ./= maximum(own, dims=1)
    @test isapprox(own, ref; rtol=1e-4, atol=1e-5)
    @test isapprox(get_spec(ch), ref; rtol=0.1, atol=0.02)
end
