using Test
using Audio911
using MAT

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
af_file(filename)   = joinpath(dirname(@__FILE__), "audioflux_files", "onset", filename)
wav_file = test_file("test.wav")
rel_l2(a, b) = sqrt(sum(abs2, vec(a) .- vec(b))) / max(sqrt(sum(abs2, b)), 1e-30)

@testset "Novelty onsets" begin
    csr = 22050
    times = collect(0.5:0.5:7.5)
    y = clicks(times; sr=csr, click_freq=1000, click_duration=0.05, length=8csr)
    st = Stft(y, csr; winsize=2048, winstep=512, type=hanning, center=true, keep_complex=true)
    for (method, kw) in ((SpectralFlux, (; p=1, positive=true, root=false)), (SpectralHfc, (;)),
                         (SpectralSd, (; positive=true)), (SpectralSf, (; positive=true)),
                         (SpectralCd, (;)), (SpectralRcd, (;)))
        env = Novelty(st; method, kw...)
        @test env isa Novelty
        @test extrema(get_data(env)) == (0, 1)
        @test get_times(env) == get_times(st)
        det = get_times(env)[onset_detect(env)]
        @test count(minimum(abs.(det .- tt)) < 0.05 for tt in times) ≥ length(times) - 1
    end
    @test Novelty(st; filter_order=3) isa Novelty
    @test_throws ArgumentError Novelty(st; filter_order=0)
    # peak picking follows librosa: the max window ends before n + post_max
    x = [0.0, 1.0, 2.0, 0.0]
    @test peak_pick(x; pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.1, wait=0) == [2, 3]
    @test peak_pick(x; pre_max=1, post_max=2, pre_avg=1, post_avg=1, delta=0.1, wait=0) == [3]
end

# obj = af.Onset(time_length=T, fre_length=257, slide_length=256, samplate=sr,
#                filter_order=<order>, novelty_type=<type>); point, evn, _, _ = obj.onset(mag, phase)
# with audioFlux's default NoveltyParam (step 1, p 1, positive, no root, sum)
@testset "onsets against audioFlux" begin
    audio = Audio911.load(wav_file; format=Float64)
    s = Stft(Frames(audio; winsize=512, winstep=256, type=hanning); spectrum=magnitude, keep_complex=true)
    m = MAT.matread(af_file("onset.mat"))
    cases = [
        ("flux_1", SpectralFlux, 1, (; p=1, positive=true, root=false)), ("hfc_1", SpectralHfc, 1, (;)),
        ("sd_1", SpectralSd, 1, (; positive=true)), ("sf_1", SpectralSf, 1, (; positive=true)),
        ("mkl_1", SpectralMkl, 1, (;)), ("pd_1", SpectralPd, 1, (;)), ("wpd_1", SpectralWpd, 1, (;)),
        ("nwpd_1", SpectralNwpd, 1, (;)), ("cd_1", SpectralCd, 1, (;)), ("rcd_1", SpectralRcd, 1, (;)),
        ("broadband_1", SpectralBroadband, 1, (;)),
        ("flux_3", SpectralFlux, 3, (; p=1, positive=true, root=false)), ("sd_5", SpectralSd, 5, (; positive=true)),
    ]
    for (key, method, order, kw) in cases
        env = Novelty(s; method, filter_order=order, kw...)
        tol = key in ("pd_1", "broadband_1", "mkl_1") ? 2e-2 : 1e-4
        @test rel_l2(get_data(env), m["evn_" * key]) < tol
        # audioFlux's peak picking rounds the windows down from 30 ms / 100 ms
        @test onset_detect(env) == Int.(vec(m["point_" * key]))
    end
end
