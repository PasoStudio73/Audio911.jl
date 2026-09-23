using Test
using Audio911
using MAT

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
af_files_dir()      = joinpath(dirname(@__FILE__), "audioflux_files", "dsp")
af_file(filename)   = joinpath(af_files_dir(), filename)

wav_file = test_file("test.wav")
rel_l2(a, b) = sqrt(sum(abs2, a .- b)) / sqrt(sum(abs2, b))
const fft = Audio911.FFTW.fft

@testset "czt, xcorr, convolve" begin
    x = sin.(0.3 .* (0:99)) .+ 0.1 .* cos.(1.7 .* (0:99))
    @test czt(x) ≈ fft(x)
    @test czt(x, 150) ≈ fft(vcat(x, zeros(50)))[1:150] atol=1e-9
    # a zoom band: the DTFT at the band's frequencies
    Z = czt(x, (0.1, 0.3); m=64)
    f = 0.1 .+ 0.2 .* (0:63) ./ 64
    @test Z ≈ [sum(x[n + 1] * cispi(-2fk * n) for n in 0:99) for fk in f]
    # a general spiral contour
    w, a = 0.999 * cis(-0.05), 1.01 * cis(0.2)       # |w| far from 1 overflows the chirp (as in SciPy)
    @test czt(x, 40, w, a) ≈ [sum(x[n + 1] * (a * w^(-k))^(-n) for n in 0:99) for k in 0:39]
    @test eltype(czt(Float32.(x))) == ComplexF32
    @test_throws ArgumentError czt(x, (0.3, 0.1))

    y = circshift(x, 7) .+ 0.01
    r = xcorr(x, y)
    @test length(r) == 199
    @test r ≈ [sum(x[n + l] * y[n] for n in max(1, 1 - l):min(100, 100 - l)) for l in -99:99]
    @test xcorr(x; normalize=true)[100] ≈ 1
    @test argmax(xcorr(x)) == 100
    @test length(xcorr(x, y[1:60])) == 199                  # the shorter input is padded
    @test eltype(xcorr(ComplexF64.(x))) == ComplexF64

    k = [1.0, 2.0, 3.0, 4.0]
    u = collect(1.0:6.0)
    full = [sum(u[i] * k[j - i + 1] for i in max(1, j - 3):min(6, j)) for j in 1:9]
    @test convolve(u, k) ≈ full
    @test convolve(u, k; mode=:same) ≈ full[3:8]            # floor(4/2) = 2 samples in
    @test convolve(u, k[1:3]; mode=:same) ≈ convolve(u, k[1:3])[2:7]
    @test convolve(u, k; mode=:valid) ≈ full[4:6]
    @test convolve(k, u; mode=:valid) == Float64[]
    @test eltype(convolve(Float32.(u), Float32.(k))) == Float32
    @test_throws ArgumentError convolve(u, k; mode=:circular)
end

@testset "weightings and utilities" begin
    @test isapprox(B_weighting(1000), 0; atol=0.02)
    @test isapprox(D_weighting(1000), 0; atol=0.02)
    @test B_weighting(1) == -80 && B_weighting(1; min_db=nothing) < -80
    @test D_weighting(10) > D_weighting(1)
    P = [1.0 0.1; 0.01 1e-12]
    @test power_to_db(P; min_db=-15, top_db=nothing) == max.(power_to_db(P; top_db=nothing), -15)
    @test minimum(amplitude_to_db(P; min_db=-30, top_db=nothing)) == -30

    X = [1.0 10.0; 2.0 10.0; 4.0 10.0]
    @test feature_scale(X) == [0.0 0.0; 1/3 0.0; 1.0 0.0]        # a constant column gives zeros
    @test feature_scale(X; dims=2)[:, 1] == [0.0, 0.0, 0.0]
    @test feature_scale([1.0, 2.0, 3.0]; method=:standard) ≈ [-1, 0, 1] ./ sqrt(2 / 3)
    @test feature_scale([1.0, 2.0, 3.0]; method=:standard, corrected=true) ≈ [-1, 0, 1]
    @test feature_scale([3.0, 1.0, 2.0]; method=:robust) == feature_scale([1.0, 2.0, 3.0]; method=:robust)[[3, 1, 2]]
    @test feature_scale([-2.0, 1.0]; method=:maxabs) == [-1.0, 0.5]
    @test feature_scale([1.0, 3.0]; method=:center) == [-1.0, 1.0]
    @test feature_scale([0.0, Inf]; method=:arctan) == [0.0, 1.0]
    @test eltype(feature_scale(Float32[1, 2])) == Float32
    @test_throws ArgumentError feature_scale(X; method=:zscore)

    mx, av, q = temporal_db([1.0, 0.1, 0.0])
    @test mx ≈ 0 atol=1e-6
    @test av ≈ (0 - 20 - 36) / 3 atol=1e-6
    @test q ≈ 2 / 3
    y = synth_f0([0.0, 0.5, 1.0], [100.0, 100.0, 100.0], 8000)
    @test length(y) == 8000
    @test y ≈ sin.(2π * 100 .* (1:8000) ./ 8000)
    @test maximum(abs, synth_f0([0.0, 1.0], [50.0, 50.0], 1000; amplitudes=[0.5, 0.5])) ≈ 0.5 atol=1e-3
    @test_throws DimensionMismatch synth_f0([0.0, 1.0], [50.0], 1000)
end

@testset "DSP and utilities against audioFlux" begin
    m = MAT.matread(af_file("dsp.mat"))
    v(k) = vec(m[k])
    # obj = af.CZT(radix2_exp=10); cztObj_czt on zero-padded buffers (see the script)
    seg = Float64.(v("czt_x"))
    @test rel_l2(czt(seg), v("czt_full")) < 5e-4            # float32 chirp phases in audioFlux
    @test rel_l2(czt(seg, (0.05, 0.2)), v("czt_band")) < 5e-4
    # r, _ = af.Xcorr().xcorr(a, b, XcorrNormalType.NONE / COEFF); xcorr(a, None, COEFF)
    a, b = Float64.(v("xc_a")), Float64.(v("xc_b"))
    @test rel_l2(xcorr(a, b), v("xc_none")) < 1e-5
    @test rel_l2(xcorr(a, b; normalize=true), v("xc_coeff")) < 1e-5
    @test rel_l2(xcorr(a; normalize=true), v("xc_auto")) < 1e-5
    # convObj_conv(obj, sig, n, k, m, &mode, &method, out): modes full/same/valid, direct and FFT
    sig = Float64.(v("conv_sig"))
    for kn in (31, 32), (mode, ms) in ((0, :full), (1, :same), (2, :valid)), method in (1, 2)
        ref = v("conv_$(kn)_$(mode)_$(method)")
        y = convolve(sig, Float64.(v("conv_k$kn")); mode=ms)
        @test length(y) == length(ref)
        @test rel_l2(y, ref) < 1e-5
    end
    # af.utils.auditory_weight_a/b/c/d(f)
    f = Float64.(v("w_f"))
    @test maximum(abs.(A_weighting.(f) .- v("w_a"))) < 0.01    # audioFlux: 12200 Hz for 12194
    @test maximum(abs.(B_weighting.(f) .- v("w_b"))) < 1e-3
    @test maximum(abs.(C_weighting.(f) .- v("w_c"))) < 0.01    # audioFlux: +0.062 dB for +0.06
    # audioFlux's D curve is the standard one with (c3 - f²)(c1 - f²) for (c3 - f²)²
    f2 = f .^ 2
    d_typo = @. 20 * (0.5 * log10(f2) - log10(8.3046305e-3^2) +
                      0.5 * (log10((1018.7^2 - f2)^2 + 1039.6^2 * f2) - log10((3136.5^2 - f2) * (1018.7^2 - f2) + 3424.0^2 * f2) -
                             log10(282.7^2 + f2) - log10(1160.0^2 + f2)))
    @test maximum(abs.(max.(d_typo, -80) .- v("w_d"))) < 1e-3
    @test maximum(abs.(D_weighting.(f) .- v("w_d"))) > 9            # the typo shifts the low band by 9.8 dB
    # af.utils.power_to_db / power_to_abs_db / mag_to_abs_db / log_compress / log10_compress
    S = Float64.(m["S"]); P = S .^ 2
    @test maximum(abs.(power_to_db(P; ref=maximum, top_db=80) .- m["db_rel"])) < 1e-4
    D = power_to_db(P; ref=1024^2, top_db=nothing, min_db=-80)
    @test maximum(abs.(D .- m["db_abs"])) < 1e-4
    @test maximum(abs.((maximum(D) .- D) .- m["db_abs_norm"])) < 1e-4
    @test maximum(abs.(amplitude_to_db(S; ref=1024, top_db=nothing, min_db=-80) .- m["db_mag"])) < 1e-4
    @test maximum(abs.(log1p.(10 .* S) .- m["log_c"])) < 1e-5
    @test maximum(abs.(log10.(1 .+ 10 .* S) .- m["log10_c"])) < 1e-5
    # af.utils.<name>_scale(M): the robust scaler agrees on sorted columns only
    M, Ms = Float64.(m["sc_M"]), Float64.(m["sc_Ms"])
    for (name, meth) in (("min_max", :minmax), ("max_abs", :maxabs), ("center", :center), ("mean", :mean), ("arctan", :arctan))
        @test maximum(abs.(feature_scale(M; method=meth) .- m["sc_" * name])) < 1e-5
    end
    @test maximum(abs.(feature_scale(Ms; method=:robust) .- m["scs_robust"])) < 1e-5
    @test maximum(abs.(feature_scale(M; method=:robust) .- m["sc_robust"])) > 1     # audioFlux's unsorted quartiles
    @test maximum(abs.(feature_scale(M; method=:standard, corrected=true) .- m["sc_stand_0"])) < 1e-5
    @test maximum(abs.(feature_scale(M; method=:standard) .- m["sc_stand_1"])) < 1e-5
    # af.utils.temproal_db(x, base=18.0)
    x = vec(get_data(Audio911.load(wav_file; format=Float64)))
    @test all(isapprox.(collect(temporal_db(x)), v("tdb"); atol=1e-4))
    # af.utils.synth_f0(times, f0, 16000, amplitudes) and without amplitudes
    y = synth_f0(v("sf_times"), v("sf_f0"), 16000; amplitudes=v("sf_amp"))
    @test length(y) == length(v("sf_y"))
    @test rel_l2(y, v("sf_y")) < 2e-3                           # float32 phase accumulation
    @test rel_l2(synth_f0(v("sf_times"), v("sf_f0"), 16000), v("sf_y1")) < 2e-3
end
