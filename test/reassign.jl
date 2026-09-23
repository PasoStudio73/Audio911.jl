using Test
using Audio911
using MAT

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
af_files_dir()      = joinpath(dirname(@__FILE__), "audioflux_files", "reassign")
af_file(filename)   = joinpath(af_files_dir(), filename)

wav_file = test_file("test.wav")
rel_l2(a, b) = sqrt(sum(abs2, a .- b)) / sqrt(sum(abs2, b))

# ---------------------------------------------------------------------------- #
#                            complex STFT and ISTFT                            #
# ---------------------------------------------------------------------------- #
@testset "complex STFT and istft" begin
    audio  = Audio911.load(wav_file; format=Float64)
    x      = vec(get_data(audio))
    frames = Frames(audio; winsize=512, winstep=128, type=hanning)
    stft   = Stft(frames)
    kept   = Stft(frames; keep_complex=true)

    # the real spectrogram does not change and the complex one is consistent
    @test get_spec(kept) == get_spec(stft)
    @test stft.cplx === nothing
    C = get_complex(stft)
    @test size(C) == size(get_spec(stft)) && eltype(C) == ComplexF64
    @test get_complex(kept) === kept.cplx
    @test get_complex(kept) ≈ C
    @test abs2.(C) ≈ get_spec(stft)
    @test abs.(C) ≈ get_spec(Stft(frames; spectrum=magnitude))
    @test get_phase(stft) ≈ angle.(C)
    @test get_spec(Stft(frames; keep_complex=true, scale=0.5)) ≈ 0.5 .* get_spec(stft)

    # perfect reconstruction with centred frames (Hann, 75 % overlap)
    for method in (:wola, :ola)
        st = Stft(audio; winsize=1024, winstep=256, center=true, keep_complex=true)
        y = istft(st; method)
        @test length(y) == length(x)
        @test isapprox(y, x; rtol=1e-8, atol=1e-10)
    end
    # zero padding (nfft > winsize) and plain frames: exact away from the ends
    st = Stft(Frames(audio; winsize=400, winstep=100); nfft=512)
    y = istft(st)
    @test length(y) == length(x)
    @test isapprox(y[400:end-600], x[400:end-600]; rtol=1e-8, atol=1e-10)
    # the matrix form
    y2 = istft(get_complex(stft), 512, 128; window=hanning)
    @test length(y2) == (get_nframes(stft) - 1) * 128 + 512
    @test isapprox(y2[513:end-512], x[513:length(y2)-512]; rtol=1e-8, atol=1e-10)
    @test length(istft(get_complex(stft), 512, 128; length=1000)) == 1000
    @test length(istft(get_complex(stft), 512, 128; length=60000)) == 60000
    @test_throws ArgumentError istft(C, 512, 128; method=:foo)
    @test_throws DimensionMismatch istft(C, 512, 128; nfft=1024)

    # Float32 stays Float32
    a32 = Audio911.load(wav_file; format=Float32)
    s32 = Stft(a32; winsize=512, winstep=128, center=true)
    @test eltype(get_complex(s32)) == ComplexF32
    y32 = istft(s32)
    @test eltype(y32) == Float32
    @test isapprox(y32, vec(get_data(a32)); rtol=1e-4, atol=1e-5)
end

# ---------------------------------------------------------------------------- #
#                                reassignment                                  #
# ---------------------------------------------------------------------------- #
@testset "Reassign" begin
    sr = 16000
    t  = (0:sr-1) ./ sr
    f0 = 1234.5
    tone = sin.(2π * f0 .* t)
    stft = Stft(Frames(tone, sr; winsize=512, winstep=128, type=hanning))
    r = Reassign(stft)
    @test r isa Reassign
    @test size(get_spec(r)) == size(get_spec(stft))
    @test get_freq(r) == get_freq(stft)
    @test get_times(r) == get_times(stft)
    @test get_frontend(r) === r && get_parent(r) === stft
    F, Tm = get_reassigned(r)
    @test size(F) == size(Tm) == size(get_spec(stft))
    # the strong cells of a stationary tone are reassigned to its frequency
    strong = get_spec(stft) .> 0.1 * maximum(get_spec(stft))
    @test all(abs.(F[strong] .- f0) .< 2)
    # and the reassigned spectrogram is sharper than the STFT
    prof_r = vec(sum(get_spec(r), dims=2)); prof_s = vec(sum(get_spec(stft), dims=2))
    @test maximum(prof_r) / sum(prof_r) > 1.5 * maximum(prof_s) / sum(prof_s)
    @test get_freq(r)[argmax(prof_r)] ≈ f0 atol=sr / 512

    # energy accumulation preserves the total energy
    re = Reassign(stft; accumulate=:energy)
    @test sum(get_spec(re)) ≈ sum(get_spec(stft))
    rm = Reassign(stft; accumulate=:energy, spectrum=magnitude)
    @test sum(get_spec(rm)) ≈ sum(sqrt.(get_spec(stft)))
    @test_throws ArgumentError get_complex(re)
    @test size(get_complex(r)) == size(get_spec(r))

    # an impulse is reassigned in time onto its frame
    imp = zeros(sr ÷ 2); imp[4001] = 1
    si = Stft(Frames(imp, sr; winsize=512, winstep=128, type=hanning))
    ri = Reassign(si; mode=:time)
    _, Ti = get_reassigned(ri)
    hit = get_spec(si) .> 1e-3 * maximum(get_spec(si))
    @test all(abs.(Ti[hit] .- 4000 / sr) .< 128 / sr)
    Fi, _ = get_reassigned(ri)
    @test Fi == repeat(collect(get_freq(si)), 1, get_nframes(si))   # frequencies untouched

    # options
    @test Reassign(stft; mode=:freq) isa Reassign
    @test Reassign(stft; order=3) isa Reassign
    @test_throws ArgumentError Reassign(stft; mode=:foo)
    @test_throws ArgumentError Reassign(stft; accumulate=:foo)
    @test_throws ArgumentError Reassign(stft; order=0)

    # the downstream chain on a reassigned spectrogram
    audio  = Audio911.load(wav_file; format=Float64)
    frames = Frames(audio; winsize=512, winstep=128, type=hanning)
    ra = Reassign(Stft(frames))
    mel = MelSpec(ra; nbands=26)
    @test size(get_data(mel)) == (length(frames), 26)
    @test size(get_data(Mfcc(mel; ncoeffs=13))) == (length(frames), 13)
    @test length(get_data(SpectralCentroid(ra))) == length(frames)
    @test size(get_spec(Chroma(ra))) == (12, length(frames))
    @test size(get_data(LinSpec(ra; freqrange=(100, 4000))), 1) == length(frames)

    # Float32
    a32 = Audio911.load(wav_file; format=Float32)
    r32 = Reassign(Stft(a32; winsize=512, winstep=128))
    @test eltype(get_spec(r32)) == Float32
    @test eltype(get_reassigned(r32)[1]) == Float32
end

# ---------------------------------------------------------------------------- #
#                             against audioFlux                                #
# ---------------------------------------------------------------------------- #
@testset "STFT, istft and reassignment against audioFlux" begin
    audio  = Audio911.load(wav_file; format=Float64)
    x      = vec(get_data(audio))
    frames = Frames(audio; winsize=512, winstep=128, type=hanning)
    stft   = Stft(frames; keep_complex=true)
    C = get_complex(stft)

    # obj = af.STFT(radix2_exp=9, window_type=HANN, slide_length=128); S = obj.stft(x)
    # y0 = obj.istft(S, method_type=0); y1 = obj.istft(S, method_type=1)
    mat = MAT.matread(af_file("stft.mat"))
    @test size(mat["stft"]) == size(C)
    @test rel_l2(C, mat["stft"]) < 1e-5
    y0 = istft(C, 512, 128; window=hanning, method=:wola)
    y1 = istft(C, 512, 128; window=hanning, method=:ola)
    @test rel_l2(y0, vec(mat["istft_wola"])) < 1e-5
    @test rel_l2(y1, vec(mat["istft_ola"])) < 1e-5

    # obj = af.Reassign(radix2_exp=9, samplate=sr, window_type=HANN, slide_length=128,
    #                   re_type=<mode>, thresh=0.001, is_padding=False)
    # obj.set_order(order); re, S = obj.reassign(x, result_type=<0|1>)
    # Rounding the reassigned bin of a cell that lands on a half bin can go
    # either way in float32 and float64, so a few cells in 10⁵ move by one
    # bin; the matrices agree to ~2e-6 relative L2 error.
    for (name, mode, order, acc) in (("all", :all, 1, :complex), ("fre", :freq, 1, :complex),
                                     ("time", :time, 1, :complex), ("all_order2", :all, 2, :complex),
                                     ("all_mag", :all, 1, :energy))
        mat = MAT.matread(af_file(name * ".mat"))
        ref = acc === :complex ? abs.(mat["re"]) : real.(mat["re"])
        r = Reassign(stft; mode, order, accumulate=acc, spectrum=magnitude)
        S = get_spec(r)
        @test size(S) == size(ref)
        @test rel_l2(S, ref) < 1e-4
        close = abs.(S .- ref) .≤ 1e-3 .* max.(abs.(ref), 1e-3 * maximum(ref))
        @test count(close) / length(close) > 0.9999
    end
end
