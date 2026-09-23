using Test
using Audio911
using MAT

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
af_file(filename)   = joinpath(dirname(@__FILE__), "audioflux_files", "spectral", filename)
wav_file = test_file("test.wav")
rel_l2(a, b) = sqrt(sum(abs2, a .- b)) / max(sqrt(sum(abs2, b)), 1e-30)

const ALL_AF = (SpectralEnergy, SpectralRms, SpectralHfc, SpectralSd, SpectralSf, SpectralMkl,
                SpectralBroadband, SpectralNovelty, SpectralEef, SpectralEer, SpectralMax,
                SpectralPeak, SpectralMean, SpectralVar)
const PHASE_AF = (SpectralPd, SpectralWpd, SpectralNwpd, SpectralCd, SpectralRcd)

@testset "audioFlux descriptors, structural" begin
    audio  = Audio911.load(wav_file; format=Float64)
    frames = Frames(audio; winsize=512, winstep=256, type=hanning)
    stft   = Stft(frames)
    mel    = MelSpec(stft; nbands=26)
    for D in (ALL_AF..., PHASE_AF...)
        x = D(stft)
        @test length(get_data(x)) == length(frames)
        @test all(isfinite, get_data(x))
        @test get_times(x) == get_times(stft)
    end
    for D in ALL_AF                                # any spectrogram
        @test length(get_data(D(mel))) == length(frames)
    end
    for D in PHASE_AF                              # need complex coefficients
        @test_throws ArgumentError D(mel)
        @test length(get_data(D(Cqt(frames; nbins=60)))) == length(frames)
    end
    # definitions on a known spectrogram
    S = get_spec(stft)
    @test get_data(SpectralMax(stft)) == vec(maximum(S, dims=1))
    @test get_data(SpectralMean(stft)) ≈ vec(sum(S, dims=1)) ./ size(S, 1)
    @test get_data(SpectralPeak(stft))[10] == get_freq(stft)[argmax(S[:, 10])]
    @test get_data(SpectralHfc(stft))[5] ≈ sum((0:size(S, 1)-1) .* S[:, 5])
    @test get_data(SpectralSd(stft))[1] == 0
    @test get_data(SpectralSf(stft))[7] ≈ sum(abs2, S[:, 7] .- S[:, 6])
    @test get_data(SpectralEnergy(stft)) ≈ vec(sum(S, dims=1)) ./ size(S, 1)       # power input
    @test get_data(SpectralEnergy(Stft(frames; spectrum=magnitude))) ≈ get_data(SpectralEnergy(stft))
    @test all(get_data(SpectralNovelty(stft; data=:number)) .≤ size(S, 1))
    @test get_data(SpectralPd(stft))[1:2] == [0, 0]
    # the MATLAB definitions are the defaults
    @test get_data(SpectralFlux(stft)) ≈ get_data(SpectralFlux(stft; p=2, step=1, positive=false, root=true, mean=false))
    @test get_data(SpectralFlux(stft; root=false)) ≈ get_data(SpectralFlux(stft)) .^ 2
    @test get_data(SpectralEntropy(stft; normalize=false)) ≈ get_data(SpectralEntropy(stft)) .* log2(size(S, 1))
    @test_throws ArgumentError SpectralNovelty(stft; method=:foo)
    @test_throws ArgumentError SpectralNovelty(stft; data=:foo)
    @test_throws ArgumentError SpectralFlux(stft; step=0)
    # Float32
    a32 = Audio911.load(wav_file; format=Float32)
    s32 = Stft(a32; winsize=512, winstep=256)
    for D in (ALL_AF..., PHASE_AF...)
        @test eltype(get_data(D(s32))) == Float32
    end
end

# obj = af.Spectral(num=257, fre_band_arr=freq); obj.set_time_length(T); obj.<feature>(mag, ...)
# on the magnitude STFT (periodic Hann 512, hop 256) and its phase, see
# test/audioflux_sources/spectral.py. Features that divide by near-silent
# bins (mkl, the novelty ratios, broadband counts) or difference raw phases
# (pd) carry the float32 rounding of the reference.
@testset "audioFlux descriptors against audioFlux" begin
    audio = Audio911.load(wav_file; format=Float64)
    s = Stft(Frames(audio; winsize=512, winstep=256, type=hanning); spectrum=magnitude, keep_complex=true)
    m = MAT.matread(af_file("spectral.mat"))
    @test rel_l2(get_spec(s), m["mag"]) < 1e-6
    cases = [
        ("flatness", SpectralFlatness(s), 1e-3), ("flux", SpectralFlux(s; root=false), 1e-4),
        ("flux_p1_pos", SpectralFlux(s; p=1, step=2, positive=true, root=false, mean=true), 1e-4),
        ("flux_exp", SpectralFlux(s), 1e-4), ("rolloff", SpectralRolloff(s; threshold=0.9), 1e-6),
        ("centroid", SpectralCentroid(s), 1e-4), ("spread", SpectralSpread(s), 1e-4),
        ("skewness", SpectralSkewness(s), 1e-4), ("kurtosis", SpectralKurtosis(s), 1e-4),
        ("entropy", SpectralEntropy(s; normalize=false), 1e-4), ("entropy_norm", SpectralEntropy(s), 1e-4),
        ("crest", SpectralCrest(s), 1e-4), ("slope", SpectralSlope(s), 1e-4),
        ("decrease", SpectralDecrease(s), 1e-4), ("band_width", SpectralBandwidth(s; normalize=false), 1e-4),
        ("rms", SpectralRms(s), 1e-4), ("energy", SpectralEnergy(s), 1e-4),
        ("energy_log", SpectralEnergy(s; log=true, gamma=10), 1e-4), ("hfc", SpectralHfc(s), 1e-4),
        ("sd", SpectralSd(s), 1e-4), ("sd_pos", SpectralSd(s; step=2, positive=true), 1e-4),
        ("sf", SpectralSf(s), 1e-4), ("mkl", SpectralMkl(s), 1e-3), ("mkl_mean", SpectralMkl(s; mean=true), 1e-3),
        ("pd", SpectralPd(s), 5e-3), ("wpd", SpectralWpd(s), 1e-4), ("nwpd", SpectralNwpd(s), 1e-4),
        ("cd", SpectralCd(s), 1e-4), ("rcd", SpectralRcd(s), 1e-4),
        ("broadband", SpectralBroadband(s; threshold=1), 1e-2),
        ("novelty", SpectralNovelty(s), 1e-4),
        ("novelty_kl_number", SpectralNovelty(s; threshold=0.001, method=:kl, data=:number), 1e-2),
        ("novelty_entropy", SpectralNovelty(s; step=2, method=:entropy), 1e-3),
        ("novelty_is", SpectralNovelty(s; method=:is), 2e-2),
        ("eef", SpectralEef(s), 1e-4), ("eef_norm", SpectralEef(s; normalize=true), 1e-4),
        ("eer", SpectralEer(s; gamma=10), 1e-4), ("max", SpectralMax(s), 1e-4),
        ("max_fre", SpectralPeak(s), 1e-6), ("mean", SpectralMean(s), 1e-4), ("var", SpectralVar(s), 1e-4),
    ]
    for (name, d, tol) in cases
        @test rel_l2(get_data(d), vec(m[name])) < tol
    end
    # audioFlux raises the signed deviation to p ≠ 2 (librosa and Audio911 use |f - c|^p)
    @test rel_l2(get_data(SpectralBandwidth(s; p=3, normalize=false)), vec(m["band_width_3"])) > 1e-3
end
