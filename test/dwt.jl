using Test
using Audio911
using MAT

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
af_files_dir()      = joinpath(dirname(@__FILE__), "audioflux_files", "dwt")
af_file(filename)   = joinpath(af_files_dir(), filename)

wav_file = test_file("test.wav")
rel_l2(a, b) = sqrt(sum(abs2, a .- b)) / sqrt(sum(abs2, b))

# ---------------------------------------------------------------------------- #
#                                 structural                                   #
# ---------------------------------------------------------------------------- #
@testset "discrete wavelets" begin
    @test length(DISCRETE_WAVELETS) == 51
    for name in DISCRETE_WAVELETS
        loD, hiD, loR, hiR = wavelet_filters(name)
        @test length(loD) == length(hiD) == length(loR) == length(hiR)
        @test iseven(length(loD))
        @test sum(loD) ≈ sqrt(2) rtol=1e-4              # low-pass: unit DC gain × √2
        @test abs(sum(hiD)) < 1e-4                      # high-pass: zero DC
    end
    # orthogonal wavelets: the reconstruction filters are the time-reversed ones
    for name in ("haar", "db4", "sym8", "coif3", "fk14", "db40")
        loD, hiD, loR, hiR = wavelet_filters(name)
        @test loR ≈ reverse(loD) atol=1e-5
        @test sum(abs2, loD) ≈ 1 rtol=1e-4
    end
    @test wavelet_filters("db1") == wavelet_filters("haar")
    @test_throws ArgumentError wavelet_filters("db11")

    x = randn(1024)
    # an orthogonal DWT preserves the energy
    for name in ("haar", "db4", "sym6", "coif2")
        c, img = dwt(x; wavelet=name, level=5)
        @test length(c) == 1024
        @test sum(abs2, c) ≈ sum(abs2, x) rtol=1e-3
        @test size(img) == (5, 1024)
    end
    c, _ = dwt(x; wavelet="haar", level=1)
    @test c[1] ≈ (x[1] + x[2]) / sqrt(2) rtol=1e-5 atol=1e-6
    @test c[513] ≈ (x[1] - x[2]) / sqrt(2) rtol=1e-5 atol=1e-6     # PyWavelets sign
    @test_throws ArgumentError dwt(randn(1000); level=4)
    @test_throws ArgumentError dwt(x; level=0)

    # the wave-packet leaves are in frequency order: a tone lands in its band
    sr = 16000
    t  = (0:4095) ./ sr
    for f0 in (700.0, 2300.0, 5900.0)
        tone = sin.(2π * f0 .* t)
        leaves, _ = wpt(tone; wavelet="db8", level=4)
        e = [sum(abs2, leaves[(k-1)*256+1:k*256]) for k in 1:16]
        @test argmax(e) == floor(Int, f0 / 500) + 1
    end
    lw, img = wpt(x; wavelet="sym4", level=3)
    @test sum(abs2, lw) ≈ sum(abs2, x) rtol=1e-3
    @test size(img) == (8, 1024)

    # the stationary transform keeps the length and splits the energy
    A, D = swt(x; wavelet="haar", level=3)
    @test size(A) == size(D) == (3, 1024)
    @test sum(abs2, A[1, :]) + sum(abs2, D[1, :]) ≈ 2 * sum(abs2, x) rtol=1e-3   # redundant by 2

    # front ends on the frame grid
    audio  = Audio911.load(wav_file; format=Float64)
    frames = Frames(audio; winsize=512, winstep=256)
    for (S, nb) in ((Dwt(frames; level=5), 6), (Wpt(frames; level=4), 16), (Swt(frames; level=3), 4))
        @test S isa DiscreteWavelet
        @test size(get_spec(S)) == (nb, length(frames))
        @test issorted(get_freq(S)) && length(get_freq(S)) == nb
        @test all(≥(0), get_spec(S))
        @test get_times(S) == get_times(Stft(frames))
        @test size(get_data(MelSpec(S; nbands=4, freqrange=(0, 8000), norm=none_norm))) == (length(frames), 4)
        @test length(get_data(SpectralCentroid(S))) == length(frames)
        @test get_nbands(S) == nb
    end
    @test Dwt(audio; winsize=512, winstep=256, wavelet="db4", level=4) isa DiscreteWavelet
    @test get_spec(Wpt(frames; level=4, spectrum=magnitude)) ≈ sqrt.(get_spec(Wpt(frames; level=4))) rtol=0.5
    a32 = Audio911.load(wav_file; format=Float32)
    @test eltype(get_spec(Dwt(Frames(a32; winsize=512, winstep=256); level=4))) == Float32
    @test eltype(dwt(Float32.(x); level=3)[1]) == Float32
end

# ---------------------------------------------------------------------------- #
#                            against audioFlux                                 #
# ---------------------------------------------------------------------------- #
# The coefficient tables are audioFlux's (six decimals), so the transforms
# agree to single precision.
@testset "DWT, WPT, SWT against audioFlux" begin
    audio = Audio911.load(wav_file; format=Float64)
    x = vec(get_data(audio))[1:2^12]

    # af.DWT(num=11, radix2_exp=12, samplate=sr, wavelet_type=..., t1=..., t2=...); coef, m = obj.dwt(x)
    for name in DISCRETE_WAVELETS
        mat = MAT.matread(af_file("dwt_" * replace(name, "." => "_") * ".mat"))
        c, img = dwt(x; wavelet=name, level=11)
        @test rel_l2(c, vec(mat["coef"])) < 1e-4
        @test rel_l2(img, mat["image"]) < 1e-4
    end
    # audioFlux 0.1.9's Python DWT wrapper shifts its arguments and always
    # applies sym4 (the fixtures above call the C library directly)
    mat = MAT.matread(af_file("dwt_wrapper_db2.mat"))
    @test rel_l2(dwt(x; wavelet="sym4", level=11)[1], vec(mat["coef"])) < 1e-4
    @test rel_l2(dwt(x; wavelet="db2", level=11)[1], vec(mat["coef"])) > 1e-2

    mat = MAT.matread(af_file("dwt_sym4_level5.mat"))
    c, img = dwt(x; wavelet="sym4", level=5)
    @test rel_l2(c, vec(mat["coef"])) < 1e-4
    @test rel_l2(img, mat["image"]) < 1e-4

    # af.WPT(num=level, radix2_exp=12, samplate=sr, wavelet_type=..., t1, t2); coef, m = obj.wpt(x)
    for (name, level) in (("sym4", 5), ("db4", 3), ("bior3.5", 4), ("haar", 6))
        mat = MAT.matread(af_file("wpt_$(replace(name, "." => "_"))_$level.mat"))
        c, img = wpt(x; wavelet=name, level)
        @test rel_l2(c, vec(mat["coef"])) < 1e-4
        @test rel_l2(img, mat["image"]) < 1e-4
    end

    # af.SWT(num=level, fft_length=4096, wavelet_type=..., t1, t2); a, d = obj.swt(x)
    for (name, level) in (("sym4", 5), ("db2", 3), ("haar", 4), ("coif3", 2))
        mat = MAT.matread(af_file("swt_$(name)_$level.mat"))
        A, D = swt(x; wavelet=name, level)
        @test rel_l2(A, mat["app"]) < 1e-4
        @test rel_l2(D, mat["det"]) < 1e-4
    end
end
