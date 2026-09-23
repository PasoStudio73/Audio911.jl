using Test
using Audio911

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
wav_file = test_file("test.wav")

# Structural tests: audioFlux declares WVD, CWD, EMD, EWT and HHT in its
# headers without an implementation, so there is no reference output.

@testset "hilbert" begin
    sr = 8000
    t = (0:799) ./ sr
    x = cos.(2π * 250 .* t)                 # 250 Hz: an integer number of periods
    z = hilbert(x)
    @test real.(z) ≈ x
    @test imag.(z) ≈ sin.(2π * 250 .* t) atol=1e-10
    @test abs.(z) ≈ ones(800) atol=1e-10
    @test hilbert(Float32[1, 2, 3]) isa Vector{ComplexF32}
    y = randn(101)                                        # odd length
    @test real.(hilbert(y)) ≈ y
    @test abs(sum(imag.(hilbert(y)))) < 1e-8 * length(y)  # the Hilbert transform has no DC
    @test isempty(hilbert(Float64[]))
end

@testset "Wigner-Ville and Choi-Williams" begin
    sr = 8000
    t = (0:4095) ./ sr
    x = sin.(2π * 1000 .* t)
    W = wvd(x[1:256]; nfft=256)
    @test size(W) == (256, 256)
    # the time marginal of the WVD is the instantaneous power |z|²
    z = hilbert(x[1:256])
    @test vec(sum(W, dims=1)) ./ 256 ≈ abs2.(z) rtol=1e-8
    # a tone concentrates at its frequency, (k-1)·sr/(2·nfft)
    @test (argmax(vec(sum(W[:, 64:192], dims=2))) - 1) * sr / 512 ≈ 1000 atol=sr / 512

    fr = Frames(x, sr; winsize=256, winstep=128, type=hanning)
    w = Wvd(fr)
    @test w isa CohenDistribution
    @test size(get_spec(w)) == (256, length(fr))
    @test get_freq(w)[2] ≈ sr / 512
    @test all(≥(0), get_spec(w))
    @test get_spec(w) == max.(get_distribution(w), 0)
    @test get_freq(w)[argmax(vec(sum(get_spec(w), dims=2)))] ≈ 1000 atol=sr / 512

    # two tones: the Choi-Williams kernel attenuates the cross term midway
    x2 = sin.(2π * 800 .* t) .+ sin.(2π * 2400 .* t)
    f2 = Frames(x2, sr; winsize=256, winstep=128, type=hanning)
    wv = get_distribution(Wvd(f2)); cw = get_distribution(Cwd(f2; sigma=1))
    k_cross = round(Int, 1600 / (sr / 512)) + 1
    k_auto  = round(Int, 800 / (sr / 512)) + 1
    ratio(D) = sum(abs, D[k_cross, 3:end-2]) / sum(abs, D[k_auto, 3:end-2])
    @test ratio(cw) < 0.5 * ratio(wv)
    @test_throws ArgumentError Cwd(f2; sigma=0)

    # downstream stages and Float32
    audio = Audio911.load(wav_file; format=Float32)
    c = Cwd(audio; winsize=256, winstep=128)
    @test eltype(get_spec(c)) == Float32
    @test size(get_data(MelSpec(c; nbands=20))) == (get_nframes(c), 20)
end

@testset "EMD and the Hilbert-Huang spectrum" begin
    sr = 4000
    t = (0:3999) ./ sr
    slow = sin.(2π * 20 .* t); fast = 0.5 .* sin.(2π * 400 .* t)
    x = slow .+ fast
    imfs, res = emd(x)
    @test size(imfs, 2) == length(x)
    @test vec(sum(imfs, dims=1)) .+ res ≈ x               # complete
    cor(a, b) = sum((a .- sum(a)/length(a)) .* (b .- sum(b)/length(b))) /
                sqrt(sum(abs2, a .- sum(a)/length(a)) * sum(abs2, b .- sum(b)/length(b)))
    @test cor(imfs[1, 200:end-200], fast[200:end-200]) > 0.95   # the fast tone first
    k = argmax([cor(imfs[i, 200:end-200], slow[200:end-200]) for i in axes(imfs, 1)])
    @test cor(imfs[k, 200:end-200], slow[200:end-200]) > 0.9
    @test size(emd(x; max_imfs=1)[1], 1) == 1
    m, r = emd(collect(1.0:100.0))                        # monotonic: no IMF
    @test size(m, 1) == 0 && r == collect(1.0:100.0)
    @test_throws ArgumentError emd(x; max_imfs=0)

    fr = Frames(x, sr; winsize=256, winstep=128)
    h = Hht(fr; nbins=129)
    @test size(get_spec(h)) == (129, length(fr))
    @test size(get_imfs(h), 2) == length(get_signal(fr))
    @test all(≥(0), get_spec(h))
    prof = vec(sum(get_spec(h), dims=2))
    top2 = sort(get_freq(h)[sortperm(prof, rev=true)[1:2]])
    @test isapprox(top2[1], 20; atol=2 * sr / 256) && isapprox(top2[2], 400; atol=2 * sr / 256)
    audio = Audio911.load(wav_file; format=Float32)
    @test eltype(get_spec(Hht(audio; winsize=512, winstep=256, max_imfs=4))) == Float32
end

@testset "empirical wavelet transform" begin
    sr = 8000
    t = (0:4095) ./ sr
    parts = [sin.(2π * f .* t) for f in (300, 1100, 2600)]
    x = sum(parts)
    comps, bounds = ewt(x, sr; nbands=3)
    @test size(comps) == (3, 4096)
    @test length(bounds) == 2 && 300 < bounds[1] < 1100 < bounds[2] < 2600
    for i in 1:3
        @test sqrt(sum(abs2, comps[i, :] .- parts[i]) / sum(abs2, parts[i])) < 0.05
    end
    # tight frame: Σ |filters|² = 1
    N = 512
    F = Audio911._ewt_filters(N, [0.3, 1.1, 2.0], 0.1)
    @test vec(sum(abs2, F, dims=1)) ≈ ones(N) atol=1e-10
    @test_throws ArgumentError ewt(x, sr; nbands=1)

    fr = Frames(x, sr; winsize=256, winstep=128)
    e = Ewt(fr; nbands=3)
    @test size(get_spec(e)) == (3, length(fr))
    @test issorted(get_freq(e))
    @test length(get_data(SpectralCentroid(e))) == length(fr)
end
