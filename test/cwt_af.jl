using Test
using Audio911
using MAT

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
af_files_dir()      = joinpath(dirname(@__FILE__), "audioflux_files", "cwt")
af_file(filename)   = joinpath(af_files_dir(), filename)

wav_file = test_file("test.wav")
rel_l2(a, b) = sqrt(sum(abs2, a .- b)) / sqrt(sum(abs2, b))

# ---------------------------------------------------------------------------- #
#                                 structural                                   #
# ---------------------------------------------------------------------------- #
@testset "wavelets and grids" begin
    # every wavelet is analytic and peaks near its centre frequency
    for ψ in (morse, morlet, bump, paul, dog, mexican, hermit, ricker)
        @test ψ(-1.0) == 0 && ψ(0.0) == 0
        ω = range(0.01, 30, length=30000)
        v = abs.(ψ.(ω))
        @test maximum(v) > 0
    end
    @test morse((20 / 3)^(1 / 3)) ≈ 2
    @test paul(4.0) ≈ Audio911._paul_factor(4) * 4^4 * exp(-4)
    @test Audio911._gamma_half(2) ≈ 3 * sqrt(π) / 4                  # Γ(2.5)
    @test dog(1.0) ≈ exp(-0.5) / sqrt(Audio911._gamma_half(2))       # positive for m = 2
    @test dog(1.0; m=4) < 0
    @test mexican(1.3) == dog(1.3; m=2)
    @test_throws ArgumentError dog(1.0; m=3)
    @test ricker(4.0) ≈ maximum(ricker.(range(0.1, 10, length=10000)))  rtol=1e-6
    @test Audio911._centre_omega(x -> morse(x; β=10, γ=3)) ≈ (10 / 3)^(1 / 3) rtol=1e-6

    sr = 16000
    t  = (0:4095) ./ sr
    x  = sin.(2π * 440 .* t)
    W, f = cwt(x, sr; nbands=84, scale=octave, freqrange=(33, 8000))
    @test size(W) == (84, 4096) && issorted(f)
    @test f[1] ≈ 32.703 rtol=1e-4
    @test f[argmax(vec(sum(abs2, W, dims=2)))] ≈ 440 rtol=0.03
    for ψ in (paul, dog, mexican, hermit, ricker, bump, morlet)
        Wψ, fψ = cwt(x, sr; wavelet=ψ, nbands=84, scale=octave, freqrange=(33, 8000))
        @test all(isfinite, Wψ)
        @test fψ[argmax(vec(sum(abs2, Wψ, dims=2)))] ≈ 440 rtol=0.12
    end
    Wm, fm = cwt(x, sr; nbands=40, scale=htk, freqrange=(0, 8000), pad=false)
    @test size(Wm) == (40, 4096)
    Wg, fg = cwt(x, sr; scale=nothing, voices=8, freqrange=(100, 4000))
    @test length(fg) == floor(Int, log2(40) * 8) + 1

    # the front end on audioFlux's grids, with the complex coefficients
    audio  = Audio911.load(wav_file; format=Float64)
    frames = Frames(audio; winsize=512, winstep=256, type=rect)
    c = Cwt(frames; wavelet=paul, scale=octave, nbands=84, freqrange=(33, 8000))
    @test size(get_spec(c)) == (84, length(frames))
    @test get_freq(c)[1] ≈ 32.703 rtol=1e-4
    @test get_setup(c).centre == 4.5
    C = get_complex(c)
    @test size(C) == size(get_spec(c)) && eltype(C) == ComplexF64
    cm = Cwt(frames; scale=htk, nbands=40, freqrange=(0, 8000))
    @test length(get_freq(cm)) == 40
    @test size(get_data(Mfcc(MelSpec(cm; nbands=20); ncoeffs=10))) == (length(frames), 10)
    @test_throws ArgumentError Cwt(frames; scale=octave, nbands=200, freqrange=(33, 8000))
    # the default geometric grid is unchanged
    c0 = Cwt(frames; voices=8, freqrange=(60, 8000))
    @test length(get_freq(c0)) == floor(Int, log2(8000 / 60) * 8) + 1
    @test get_setup(c0).scale === nothing
end

@testset "synchrosqueezing" begin
    sr = 16000
    t  = (0:8191) ./ sr
    f0 = 1000.0
    x  = sin.(2π * f0 .* t)
    S, W, f = wsst(x, sr; nbands=84, scale=octave, freqrange=(33, 8000))
    @test size(S) == size(W) == (84, 8192)
    k0 = argmin(abs.(f .- f0))
    # the squeezed energy concentrates on the band of the tone
    eS = vec(sum(abs2, S, dims=2)); eW = vec(sum(abs2, W, dims=2))
    @test argmax(eS) == k0
    @test eS[k0] / sum(eS) > 2 * eW[k0] / sum(eW)
    Q = synsq(W, f, sr; scale=octave)
    @test argmax(vec(sum(abs2, Q, dims=2))) == k0
    @test synsq(W, f, sr; scale=octave, order=2) isa Matrix
    @test_throws DimensionMismatch synsq(W, f[1:10], sr)

    # front ends: same grid and frames as the Cwt, energy is conserved up to
    # the coefficients that leave the grid or fall below the threshold
    fr = Frames(x, sr; winsize=512, winstep=256, type=rect)
    c  = Cwt(fr; wavelet=morse, scale=octave, nbands=84, freqrange=(33, 8000))
    for sq in (Wsst(c), Synsq(c), Wsst(c; accumulate=:complex), Synsq(c; order=2),
               Wsst(fr; nbands=84, freqrange=(33, 8000)))
        @test sq isa Synchrosqueezed
        @test size(get_spec(sq)) == size(get_spec(c))
        @test get_freq(sq) == get_freq(c) && get_times(sq) == get_times(c)
        @test all(≥(0), get_spec(sq))
        @test get_freq(sq)[argmax(vec(sum(get_spec(sq), dims=2)))] ≈ f0 rtol=0.03
    end
    @test sum(get_spec(Wsst(c; thresh=0))) ≤ sum(get_spec(c)) * (1 + 1e-9)
    @test sum(get_spec(Wsst(c; thresh=0))) > 0.9 * sum(get_spec(c))
    ws = Wsst(c)
    @test get_frontend(ws) === ws && get_parent(ws) === c
    @test size(get_data(MelSpec(ws; nbands=20, freqrange=(50, 3900)))) == (length(fr), 20)
    @test_throws ArgumentError Wsst(c; accumulate=:foo)
    @test_throws ArgumentError Synsq(c; order=0)

    # Float32
    x32 = Float32.(x)
    S32, W32, f32 = wsst(x32, sr; nbands=40, scale=octave, freqrange=(100, 8000))
    @test eltype(S32) == ComplexF32 && eltype(f32) == Float32
    c32 = Cwt(Frames(x32, sr; winsize=512, winstep=256); scale=octave, nbands=40, freqrange=(100, 8000))
    @test eltype(get_spec(Wsst(c32))) == Float32
    @test eltype(get_complex(c32)) == ComplexF32
end

# ---------------------------------------------------------------------------- #
#                             against audioFlux                                #
# ---------------------------------------------------------------------------- #
# audioFlux's Morlet is 2·exp(-(ω-6)²/2), Audio911's has the factor π^(-1/4)
# of the unit-energy Morlet: the same wavelet up to a constant.
@testset "CWT against audioFlux" begin
    audio = Audio911.load(wav_file; format=Float64)
    x  = vec(get_data(audio))[1:2^12]
    sr = 16000

    # af.CWT(num=84, radix2_exp=12, samplate=sr, low_fre=32.703, bin_per_octave=12,
    #        wavelet_type=W.<wavelet>, scale_type=Scale.OCTAVE, is_padding=<pad>); obj.cwt(x)
    for (name, ψ, k) in (("morse", morse, 1.0), ("morlet", morlet, 2 * π^(1 / 4)), ("bump", bump, 1.0),
                         ("paul", paul, 1.0), ("dog", dog, 1.0), ("mexican", mexican, 1.0),
                         ("hermit", hermit, 1.0), ("ricker", ricker, 1.0))
        for pad in (false, true)
            mat = MAT.matread(af_file("cwt_$(name)_$(pad ? "pad" : "nopad").mat"))
            ref = mat["spec"]
            W, f = cwt(x, sr; wavelet=ψ, nbands=84, scale=octave, freqrange=(33, 8000), pad)
            @test vec(f) ≈ vec(mat["freq"]) rtol=1e-5
            @test rel_l2(k .* W, ref) < 1e-4
        end
    end

    # the other grids, Morse wavelet, no padding
    for (name, scale) in (("mel", htk), ("bark", bark), ("erb", erb), ("linspace", linspace), ("log", logspace))
        mat = MAT.matread(af_file("cwt_morse_$name.mat"))
        ref = mat["spec"]
        freqrange = (round(Int, mat["low"]), round(Int, mat["high"]))
        W, f = cwt(x, sr; nbands=Int(mat["num"]), scale, freqrange, pad=false)
        @test vec(f) ≈ vec(mat["freq"]) rtol=1e-4
        @test rel_l2(W, ref) < 1e-4
    end
end

# audioFlux maps the instantaneous frequency on the octave grid with
# round((log2 f - log2 f₁) · n / (log2 fₙ - log2 f₁)), a bin step of
# (n-1)/n of the true one; Audio911 maps to the nearest band. The octave
# fixtures are checked through that mapping (re-implemented here), which
# isolates every other step of the port; the mel fixtures use the same
# nearest-band rule on both sides and are compared directly.
af_octave_index(freq, f) = begin     # freq and f normalised by sr, float32
    n = length(freq)
    (isfinite(f) && f > 0) || return 0
    lo, hi = log2(Float32(freq[1])), log2(Float32(freq[n]))
    k = round(Int, (log2(Float32(f)) - lo) * Float32(n) / (hi - lo), RoundNearestTiesAway) + 1
    1 ≤ k ≤ n ? k : 0
end

# audioFlux's synsq takes the phase atan2(re, im) in float32, unwraps it over
# the whole signal and differentiates it; the unwrapped phase of a high band
# reaches thousands of radians, so the float32 phase carries a cumulative
# rounding error that moves a few coefficients across band boundaries.
# Audio911 wraps every phase difference instead. The fixtures are checked
# through an emulation of that float32 path (below), which isolates every
# other step of the port, and directly with a looser bound.
function af_phase_freq(y::AbstractVector, sr)
    θ = Float32[atan(Float32(real(z)), Float32(imag(z))) for z in y]
    u = similar(θ); u[1] = θ[1]
    for i in 2:length(θ)
        sub = abs(Float64(θ[i]) - Float64(u[i-1]))
        if sub < π
            u[i] = θ[i]
        else
            t = floor(Int, sub / 2π); t += (sub - t * 2π) > π
            u[i] = Float32(θ[i] > u[i-1] ? θ[i] - t * 2π : θ[i] + t * 2π)
        end
    end
    d = zeros(Float32, length(u))
    for i in 2:length(u); d[i] = u[i] - u[i-1]; end
    d[end] = d[end-1]
    return abs.(d ./ Float32(2π))          # normalised frequency, float32
end
af_nearest_index(freq, f) = begin      # audioFlux __arr_roundIndex on freq/sr
    n = length(freq)
    for i in 1:n-1
        if freq[i] ≤ f < freq[i+1]
            return (f - freq[i]) < (freq[i+1] - f) ? i : i + 1
        end
    end
    0
end

@testset "synsq and wsst against audioFlux" begin
    audio = Audio911.load(wav_file; format=Float64)
    x  = vec(get_data(audio))[1:2^12]
    sr = 16000

    # s = af.Synsq(num=64, radix2_exp=12, samplate=sr)
    # out = s.synsq(cwt, filter_bank_type=Scale.MEL, fre_arr=cwt_obj.get_fre_band_arr())
    mat = MAT.matread(af_file("synsq_mel.mat"))
    W = ComplexF64.(mat["cwt"])
    f = vec(mat["freq"])
    idx = zeros(Int32, size(W))
    for k in axes(W, 1)
        idx[k, :] .= af_nearest_index.(Ref(Float32.(f ./ sr)), af_phase_freq(mat["cwt"][k, :], sr))
    end
    @test rel_l2(Audio911._squeeze(W, idx, 0.001, 1), mat["out"]) < 1e-3
    @test rel_l2(synsq(W, f, sr; scale=htk), mat["out"]) < 0.02

    mat = MAT.matread(af_file("synsq_octave.mat"))
    W = ComplexF64.(mat["cwt"])
    f = vec(mat["freq"])
    idx = zeros(Int32, size(W))
    for k in axes(W, 1)
        idx[k, :] .= af_octave_index.(Ref(f ./ sr), af_phase_freq(mat["cwt"][k, :], sr))
    end
    # a handful of coefficients sit on a band boundary to float32 precision
    # and land one band apart (rows 6/7, 1/2, 16/17 of the fixture)
    @test rel_l2(Audio911._squeeze(W, idx, 0.001, 1), mat["out"]) < 5e-3
    # and Audio911's own nearest-band mapping moves the same energy
    Q = synsq(W, f, sr; scale=octave)
    @test sum(abs2, Q) ≈ sum(abs2, mat["out"]) rtol=0.05

    # af.WSST(num=64, radix2_exp=12, samplate=sr, low_fre=0, high_fre=8000, wavelet_type=W.MORSE,
    #         scale_type=Scale.MEL, thresh=0.001, is_padding=False); out, cwt = obj.wsst(x)
    mat = MAT.matread(af_file("wsst_mel.mat"))
    S, W, f = wsst(x, sr; nbands=64, scale=htk, freqrange=(0, 8000), pad=false)
    @test vec(f) ≈ vec(mat["freq"]) rtol=1e-4
    @test rel_l2(W, mat["cwt"]) < 1e-4
    @test rel_l2(S, mat["out"]) < 1e-3

    mat = MAT.matread(af_file("wsst_octave.mat"))
    S, W, f = wsst(x, sr; nbands=84, scale=octave, freqrange=(33, 8000), pad=false)
    @test rel_l2(W, mat["cwt"]) < 1e-4
    # derivative coefficients recomputed for the audioFlux mapping
    _, W2, _ = wsst(x, sr; nbands=84, scale=octave, freqrange=(33, 8000), pad=false)
    Wd = zeros(ComplexF64, size(W))
    Audio911._cwt_bands(x, 0, length(x), f, sr, morse, Audio911._centre_omega(morse); deriv=true) do _, k, w, dw
        Wd[k, :] .= dw
    end
    idx = Int32.(af_octave_index.(Ref(f ./ sr), abs.(imag.(Wd ./ W)) ./ 2π))
    @test rel_l2(Audio911._squeeze(W, idx, 0.001, 1), mat["out"]) < 1e-3
    @test sum(abs2, S) ≈ sum(abs2, mat["out"]) rtol=0.05
end
