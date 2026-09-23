using Test
using Audio911
using MAT

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
af_files_dir()      = joinpath(dirname(@__FILE__), "audioflux_files", "transforms")
af_file(filename)   = joinpath(af_files_dir(), filename)

wav_file = test_file("test.wav")

# relative comparison against a float32 reference: every entry within
# `rtol` of the reference, or within `atol` × the reference maximum
approx_ref(a, b; rtol=1e-3, atol=1e-4) = isapprox(a, b; rtol, atol=atol * maximum(abs, b))

# ---------------------------------------------------------------------------- #
#                                 structural                                   #
# ---------------------------------------------------------------------------- #
@testset "pooled front ends" begin
    audio  = Audio911.load(wav_file; format=Float64)
    frames = Frames(audio; winsize=512, winstep=256, type=hanning)
    stft   = Stft(frames)

    fes = (
        Pwt(frames; nbands=60, scale=octave, freqrange=(33, 8000)),
        Pwt(frames; nbands=30, scale=htk, style=hanning, norm=bandwidth),
        St(frames; freqrange=(0, 400)),
        Fst(frames; freqrange=(0, 4000)),
        Nsgt(frames; nbands=60, scale=octave),
        Nsgt(frames; nbands=30, scale=htk, style=hamming, norm=none_norm, standard=true),
    )
    for fe in fes
        @test get_spec(fe) === get_data(fe)
        @test size(get_spec(fe), 2) == length(frames)
        @test length(get_freq(fe)) == get_nbins(fe)
        @test issorted(get_freq(fe))
        @test get_sr(fe) == 16000
        @test get_spectrum(fe) === power
        @test all(>=(0), get_spec(fe))
        @test all(isfinite, get_spec(fe))
        @test get_frames(fe) === frames && get_parent(fe) === frames
        @test get_step(fe) == 256 && get_winsize(fe) == 512
        @test get_times(fe) == get_times(stft)
        @test get_window(fe) === nothing && get_winnorm(fe) == 1
        C = get_complex(fe)
        @test size(C) == size(get_spec(fe))
        @test eltype(C) == ComplexF64
        # the downstream chain
        hi = min(round(Int, get_freq(fe)[end]), 7000)
        mel = MelSpec(fe; nbands=10, freqrange=(max(50, round(Int, get_freq(fe)[1]) + 1), hi))
        @test size(get_data(mel)) == (length(frames), 10)
        @test size(get_data(Mfcc(mel; ncoeffs=5))) == (length(frames), 5)
        for D in (SpectralCentroid, SpectralFlatness, SpectralFlux, SpectralRolloff, SpectralEntropy)
            @test length(get_data(D(fe))) == length(frames)
        end
        @test size(get_spec(Chroma(fe))) == (12, length(frames))
        @test length(get_data(OnsetStrength(fe))) == length(frames)
    end
    @test Pwt(frames; spectrum=magnitude) isa Pwt
    @test St(frames; freqrange=(0, 200), spectrum=magnitude) isa St
    @test get_nbands(fes[1]) == 60
    @test length(get_cells(fes[5])) == 60
    @test get_lengths(fes[5]) == length.(get_cells(fes[5]))
    @test all(get_lengths(fes[5]) .≥ 3)
    @test get_lengths(fes[5])[end] > get_lengths(fes[5])[1]       # wider bands, more samples
    @test_throws ArgumentError Pwt(frames; spectrum=sum)
    @test_throws ArgumentError Nsgt(frames; style=point)
    @test_throws ArgumentError Nsgt(frames; norm=area)
    @test_throws ArgumentError St(frames; factor=0)

    # whole-signal forms
    x = vec(get_data(audio))
    Y, f = pwt(x, 16000; nbands=20, scale=htk)
    @test size(Y) == (20, length(x)) && length(f) == 20
    S = st(x[1:1024]; min_index=1, max_index=64)
    @test size(S) == (64, 1024)
    @test all(st(x[1:1024]; min_index=0, max_index=0) .≈ sum(x[1:1024]) / 1024)
    F = fst(x[1:1024]; min_index=1, max_index=512)
    @test size(F) == (512, 1024)
    @test_throws ArgumentError fst(x[1:1000])
    cells, cf, lens = nsgt(x[1:4096], 16000; nbands=24, scale=octave, freqrange=(110, 8000))
    @test length(cells) == 24 && length(cf) == 24 && lens == length.(cells)
    M = nsgt_matrix(cells, lens, 4096 / 16000, maximum(lens))
    @test size(M) == (24, maximum(lens))

    # a pure tone lands on its band in every transform
    sr = 16000
    t  = (0:sr-1) ./ sr
    tone = sin.(2π * 440 .* t)
    fr = Frames(tone, sr; winsize=1024, winstep=512, type=rect)
    for fe in (Pwt(fr; nbands=84, scale=octave, freqrange=(33, 8000)),
               Nsgt(fr; nbands=84, scale=octave, freqrange=(33, 8000)),
               St(fr; freqrange=(200, 800)))
        prof = vec(sum(get_spec(fe), dims=2))
        @test isapprox(get_freq(fe)[argmax(prof)], 440; rtol=0.06)
    end
    # the fast S-transform holds one value over every dyadic frequency block:
    # the tone lies inside the block of maximal energy
    fe = Fst(fr; freqrange=(100, 2000))
    prof = vec(sum(get_spec(fe), dims=2))
    top = get_freq(fe)[prof .≥ maximum(prof) * (1 - 1e-9)]
    @test minimum(top) ≤ 440 ≤ maximum(top)
    @test maximum(top) / minimum(top) < 2.1

    # the whole-signal s-transform of a tone concentrates at the tone's bin
    N = 4096
    tt = (0:N-1) ./ sr
    y = sin.(2π * 1000 .* tt)
    S = st(y; min_index=1, max_index=N ÷ 2)
    @test argmax(vec(sum(abs2, S, dims=2))) == round(Int, 1000 * N / sr)

    # Float32 stays Float32
    a32 = Audio911.load(wav_file; format=Float32)
    f32 = Frames(a32; winsize=512, winstep=256)
    for fe in (Pwt(f32; nbands=20, scale=htk), St(f32; freqrange=(0, 200)), Fst(f32; freqrange=(0, 500)),
               Nsgt(f32; nbands=20, scale=htk))
        @test eltype(get_spec(fe)) == Float32
        @test eltype(get_freq(fe)) == Float32
        @test eltype(get_complex(fe)) == ComplexF32
    end
    @test eltype(get_cells(Nsgt(f32; nbands=20, scale=htk))[1]) == ComplexF32
end

# ---------------------------------------------------------------------------- #
#                            against audioFlux                                 #
# ---------------------------------------------------------------------------- #
@testset "transforms against audioFlux" begin
    audio = Audio911.load(wav_file; format=Float64)
    x  = vec(get_data(audio))
    sr = 16000
    x14 = x[1:2^14]
    x12 = x[1:2^12]

    # obj = af.PWT(num=84, radix2_exp=14, samplate=sr, low_fre=32.703, high_fre=8000, bin_per_octave=12,
    #              scale_type=OCTAVE, style_type=SLANEY, normal_type=NONE, is_padding=False); obj.pwt(x14)
    mat = MAT.matread(af_file("pwt_octave.mat"))
    Y, f = pwt(x14, sr; nbands=84, scale=octave, style=triangular, norm=none_norm, freqrange=(33, 8000))
    @test vec(f) ≈ vec(mat["freq"]) rtol=1e-4
    @test approx_ref(Y, mat["spec"])

    # num=40, MEL scale, HANN style, BAND_WIDTH norm, 0-8000 Hz
    mat = MAT.matread(af_file("pwt_mel_hann.mat"))
    Y, f = pwt(x14, sr; nbands=40, scale=htk, style=hanning, norm=bandwidth, freqrange=(0, 8000))
    @test vec(f) ≈ vec(mat["freq"]) rtol=1e-4
    @test approx_ref(Y, mat["spec"])

    # obj = af.ST(radix2_exp=12, min_index=1, max_index=1024, samplate=sr, factor=1., norm=1.); obj.st(x12)
    mat = MAT.matread(af_file("st.mat"))
    S = st(x12; min_index=1, max_index=1024)
    @test approx_ref(S, mat["spec"])
    # factor=0.5, norm=1.2, min_index=0, max_index=300
    mat = MAT.matread(af_file("st_factor.mat"))
    S = st(x12; min_index=1, max_index=300, factor=0.5, norm=1.2)
    @test approx_ref(S, mat["spec"])

    # obj = af.FST(radix2_exp=12, min_index=1, max_index=1024, samplate=sr); obj.fst(x12)
    mat = MAT.matread(af_file("fst.mat"))
    F = fst(x12; min_index=1, max_index=1024)
    @test approx_ref(F, mat["spec"])
    mat = MAT.matread(af_file("fst_all.mat"))
    F = fst(x12; min_index=1, max_index=2047)
    @test approx_ref(F, mat["spec"])

    # obj = af.NSGT(num=84, radix2_exp=14, samplate=sr, low_fre=32.703, high_fre=8000, bin_per_octave=12,
    #               min_len=3, nsgt_filter_bank_type=EFFICIENT, scale_type=OCTAVE,
    #               style_type=HANN, normal_type=BAND_WIDTH); obj.nsgt(x14)
    mat = MAT.matread(af_file("nsgt_octave.mat"))
    cells, f, lens = nsgt(x14, sr; nbands=84, scale=octave, style=hanning, norm=bandwidth, freqrange=(33, 8000))
    @test vec(f) ≈ vec(mat["freq"]) rtol=1e-4
    @test lens == vec(Int.(mat["lengths"]))
    M = nsgt_matrix(cells, lens, 2^14 / sr, Int(mat["max_len"]))
    @test approx_ref(M, mat["spec"])

    # num=40, MEL, STANDARD bank, HAMM style, NONE norm
    mat = MAT.matread(af_file("nsgt_mel_standard.mat"))
    cells, f, lens = nsgt(x14, sr; nbands=40, scale=htk, style=hamming, norm=none_norm, freqrange=(0, 8000), standard=true)
    @test vec(f) ≈ vec(mat["freq"]) rtol=1e-4
    @test lens == vec(Int.(mat["lengths"]))
    M = nsgt_matrix(cells, lens, 2^14 / sr, Int(mat["max_len"]))
    @test approx_ref(M, mat["spec"])
end
