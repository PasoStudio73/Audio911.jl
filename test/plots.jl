using Test
using Audio911
using RecipesBase

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
wav_file = test_file("test.wav")

# apply a recipe without a plotting backend and return the series data;
# Plots normally provides is_key_supported, so stub it for the backend-free run
RecipesBase.is_key_supported(::Symbol) = true
recipe(x; kw...) = RecipesBase.apply_recipe(Dict{Symbol,Any}(kw...), x)

@testset "recipes" begin
    audio  = Audio911.load(wav_file; format=Float64)
    frames = Frames(audio; winsize=512, winstep=256, type=hamming)
    stft   = Stft(frames)
    cwt    = Cwt(frames; voices=6, freqrange=(100, 4000))
    mel    = MelSpec(stft; nbands=20)
    erb    = ErbSpec(stft; nbands=20)
    mfcc   = Mfcc(mel; ncoeffs=13)
    delta  = Delta(mfcc)
    n      = get_nframes(stft)

    rd = recipe(audio)
    @test length(rd) == 1 && length(rd[1].args[1]) == length(audio)

    rd = recipe(frames)
    @test length(rd) == 2

    for s in (stft, cwt, mel, erb, LinSpec(stft), BarkSpec(stft; nbands=20), get_harmonic(Hpss(stft)))
        rd = recipe(s)
        @test length(rd) == 1
        t, f, z = rd[1].args
        @test length(t) == n && length(f) == get_nbins(s) && size(z) == (get_nbins(s), n)
        @test maximum(z) ≈ 0 && minimum(z) ≥ -80          # dB relative to the maximum
        @test rd[1].plotattributes[:seriestype] == :heatmap
    end
    rd = recipe(stft; db=false)
    @test rd[1].args[3] === get_spec(stft)
    rd = recipe(stft; freq_scale=:log10)
    @test all(>(0), rd[1].args[2]) && rd[1].plotattributes[:yscale] == :log10
    rd = recipe(Stft(frames; spectrum=magnitude))
    @test maximum(rd[1].args[3]) ≈ 0

    for s in (Chroma(stft), Tonnetz(stft), SpectralContrast(stft; nbands=5), PolyFeatures(stft),
              Tempogram(OnsetStrength(mel); win_length=32))
        rd = recipe(s)
        @test size(rd[1].args[3]) == size(get_spec(s))
    end

    for fb in (get_fbank(mel), get_fbank(erb), chroma_fbank(stft))
        rd = recipe(fb)
        @test length(rd) == size(get_data(fb), 1)
    end

    for c in (mfcc, delta, Gtcc(erb; ncoeffs=10))
        rd = recipe(c)
        @test size(rd[1].args[3]) == size(get_spec(c))
    end

    for d in (SpectralCentroid(stft), Rms(frames), Zcr(frames), Pitch(frames), OnsetStrength(mel))
        rd = recipe(d)
        @test length(rd) == 1 && length(rd[1].args[2]) == n
    end

    rd = recipe(Hpss(stft))
    @test length(rd) == 2
end

@testset "recipes: audioFlux ports" begin
    audio  = Audio911.load(wav_file; format=Float64)
    frames = Frames(audio; winsize=512, winstep=256)
    n = length(frames)
    for s in (Cqt(frames; nbins=60), Pwt(frames; nbands=40, scale=octave),
              Nsgt(frames; nbands=40, scale=octave), St(frames; freqrange=(0, 300)),
              Fst(frames; freqrange=(0, 2000)), MelSpec(Stft(frames); nbands=30, scale=erb, style=hanning),
              Reassign(Stft(frames)))
        rd = recipe(s)
        @test length(rd) == 1
        t, f, z = rd[1].args
        @test length(t) == n && size(z, 2) == n && size(z, 1) == length(f)
        @test maximum(z) ≈ 0
        @test rd[1].plotattributes[:seriestype] == :heatmap
    end
    @test recipe(Cqt(frames; nbins=60))[1].plotattributes[:yscale] == :log10
    @test !haskey(recipe(Cqt(frames; nbins=60); freq_scale=:linear)[1].plotattributes, :yscale)
    @test size(recipe(Chroma(Cqt(frames; nbins=60)))[1].args[3]) == (12, n)
    fb = cqt_chroma_fbank(Cqt(frames; nbins=60))
    @test length(recipe(fb)) == 12
end

# one end-to-end check with a real backend (GR, headless)
ENV["GKSwstype"] = "100"
using Plots
@testset "Plots" begin
    audio = Audio911.load(wav_file; format=Float32)
    stft  = Stft(audio; winsize=512, winstep=256)
    p = plot(stft)
    @test p isa Plots.Plot
    @test plot(MelSpec(stft; nbands=20); freq_scale=:log10) isa Plots.Plot
    @test plot(Mfcc(MelSpec(stft; nbands=20); ncoeffs=13)) isa Plots.Plot
    @test plot(SpectralCentroid(stft)) isa Plots.Plot
    @test plot(audio) isa Plots.Plot
    @test plot(get_fbank(MelSpec(stft; nbands=20))) isa Plots.Plot
    @test plot(Hpss(stft)) isa Plots.Plot
    @test plot(Cqt(audio; winsize=512, winstep=256, nbins=60)) isa Plots.Plot
    @test plot(Nsgt(audio; winsize=512, winstep=256, nbands=40, scale=octave)) isa Plots.Plot
end
