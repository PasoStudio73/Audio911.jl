using Test
using Audio911
using LinearAlgebra: I

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)

wav_file = test_file("test.wav")

# ---------------------------------------------------------------------------- #
#                       interchangeable front ends                             #
# ---------------------------------------------------------------------------- #
@testset "front-end interface" begin
    audio  = Audio911.load(wav_file; format=Float64)
    frames = Frames(audio; winsize=512, winstep=256, type=hamming)
    stft   = Stft(frames)
    cwt    = Cwt(frames; voices=8, freqrange=(60, 8000))

    for fe in (stft, cwt)
        @test get_spec(fe) === get_data(fe)
        @test size(get_spec(fe), 2) == length(frames)
        @test length(get_freq(fe)) == get_nbins(fe)
        @test issorted(get_freq(fe))
        @test get_sr(fe) == 16000
        @test get_spectrum(fe) === power
        @test all(>=(0), get_spec(fe))
        @test get_frames(fe) === frames
        @test get_step(fe) == 256 && get_winsize(fe) == 512
        @test length(get_times(fe)) == get_nframes(fe)
        @test get_times(fe)[1] ≈ 256 / 16000
    end
    @test get_window(stft) == get_window(frames)
    @test get_window(cwt) === nothing
    @test get_winnorm(cwt) == 1
    @test get_winnorm(stft) ≈ 1 / sum(get_window(frames))^2
    @test get_nbins(cwt) == floor(Int, log2(8000 / 60) * 8) + 1

    # a wavelet front end feeds every downstream stage unchanged
    mel_s = MelSpec(stft; nbands=26, freqrange=(100, 7000))
    mel_c = MelSpec(cwt;  nbands=26, freqrange=(100, 7000))
    @test size(get_data(mel_c)) == size(get_data(mel_s))
    @test get_freq(mel_c) ≈ get_freq(mel_s)
    @test get_parent(mel_c) === cwt
    @test get_frontend(mel_c) === cwt

    bark_c = BarkSpec(cwt; nbands=20, freqrange=(100, 7000))
    erb_c  = ErbSpec(cwt; nbands=20, freqrange=(100, 7000))
    lin_c  = LinSpec(cwt; freqrange=(100, 4000))
    @test size(get_data(bark_c), 2) == 20
    @test size(get_data(erb_c), 2) == 20
    @test all(100 .<= get_freq(lin_c) .<= 4000)
    @test size(get_data(lin_c), 1) == length(frames)

    mfcc_c = Mfcc(mel_c; ncoeffs=13)
    gtcc_c = Gtcc(erb_c; ncoeffs=13)
    d1, d2 = DeltaDelta = (Delta(mfcc_c), Delta(Delta(mfcc_c)))
    @test size(get_data(mfcc_c)) == (length(frames), 13)
    @test size(get_data(gtcc_c)) == (length(frames), 13)
    @test size(get_data(d2)) == size(get_data(mfcc_c))
    @test get_frontend(d2) === cwt
    @test get_times(d2) == get_times(cwt)

    for D in (SpectralCentroid, SpectralCrest, SpectralDecrease, SpectralEntropy,
              SpectralFlatness, SpectralFlux, SpectralKurtosis, SpectralRolloff,
              SpectralSkewness, SpectralSlope, SpectralSpread, SpectralBandwidth)
        x = D(cwt)
        @test length(get_data(x)) == length(frames)
        @test all(isfinite, get_data(x))
        y = D(mel_s)                 # descriptors also run on band spectrograms
        @test length(get_data(y)) == length(frames)
    end
    @test get_data(SpectralBandwidth(lin_c)) ≈ get_data(SpectralSpread(lin_c))

    # a scalogram of a pure tone peaks at the tone's scale
    sr = 16000
    t  = (0:sr-1) ./ sr
    tone = sin.(2π * 1000 .* t)
    c = Cwt(tone, sr; winsize=512, winstep=256, voices=16, freqrange=(100, 4000))
    peak = get_freq(c)[argmax(vec(sum(get_spec(c), dims=2)))]
    @test isapprox(peak, 1000; rtol=0.05)
    @test get_scales(c)[1] > get_scales(c)[end]

    # filterbanks designed on the wrong grid are rejected
    fb = auditory_fbank(stft; nbands=26)
    @test_throws DimensionMismatch MelSpec(cwt, fb)
    @test_throws ArgumentError Cwt(frames; freqrange=(100, 9000))
    @test_throws ArgumentError Cwt(frames; voices=0)

    # other wavelets
    for w in (morse, bump)
        cw = Cwt(frames; wavelet=w, voices=4, freqrange=(100, 4000))
        @test all(isfinite, get_spec(cw))
        @test get_nframes(cw) == length(frames)
    end
    @test Cwt(frames; spectrum=magnitude) isa Cwt
end

# ---------------------------------------------------------------------------- #
#                                   frames                                     #
# ---------------------------------------------------------------------------- #
@testset "frames options" begin
    audio = Audio911.load(wav_file; format=Float64)
    x = vec(get_data(audio))
    n = length(x)

    f = Frames(audio; winsize=400, winstep=160)
    @test length(f) == fld(n - 400, 160) + 1
    @test get_data(f)[:, 2] == x[161:560]
    @test get_winframes(f) ≈ get_data(f) .* get_window(f)
    @test get_offset(f) == 0

    # librosa-style centering: 1 + N ÷ hop frames, frame i centred on (i-1)*hop
    fc = Frames(audio; winsize=512, winstep=256, center=true)
    @test length(fc) == 1 + fld(n, 256)
    @test get_offset(fc) == -256
    @test get_data(fc)[257:end, 1] == x[1:256]
    @test get_data(fc)[1:256, 1] == zeros(256)
    @test get_times(Stft(fc))[1] ≈ 0
    fr = Frames(audio; winsize=512, winstep=256, center=true, pad_mode=:reflect)
    @test get_data(fr)[256:-1:1, 1] == x[2:257]
    fe = Frames(audio; winsize=512, winstep=256, center=true, pad_mode=:edge)
    @test all(get_data(fe)[1:256, 1] .== x[1])
    @test_throws ArgumentError Frames(audio; center=true, pad_mode=:bogus)

    # per-frame pre-emphasis and dc removal
    fp = Frames(audio; winsize=400, winstep=160, preemph=0.97)
    raw = x[1:400]
    @test get_data(fp)[1, 1] ≈ raw[1] * (1 - 0.97)
    @test get_data(fp)[2:end, 1] ≈ raw[2:end] .- 0.97 .* raw[1:end-1]
    fd = Frames(audio; winsize=400, winstep=160, dc_removal=true)
    @test abs(sum(get_data(fd)[:, 1])) < 1e-10
    @test get_energy(fd)[1] ≈ sum(abs2, raw .- sum(raw) / 400)
    @test get_energy(f)[1] ≈ sum(abs2, raw)

    # windows: periodic == symmetric of length n+1 truncated, povey available
    @test get_window(Frames(audio; winsize=417, type=hamming, periodic=true)) ≈ hamming(418)[1:417]
    @test get_window(Frames(audio; winsize=417, type=hamming, periodic=false)) ≈ hamming(417)
    w = get_window(Frames(audio; winsize=400, type=povey, periodic=false))
    @test w ≈ hanning(400) .^ 0.85
    @test_throws ArgumentError Frames(audio; type=sin)
    @test_throws ArgumentError Frames(audio; winsize=512, winstep=600)
    @test_throws ArgumentError Frames(audio; preemph=1.5)

    # signal-level pre-emphasis and its inverse
    y = preemphasis(x; coef=0.97)
    @test y[1] ≈ x[1] * (1 - 0.97)
    @test y[2:end] ≈ x[2:end] .- 0.97 .* x[1:end-1]
    @test deemphasis(y; coef=0.97, zi=x[1]) ≈ x
    @test preemphasis(x; coef=0.97, zi=0)[1] == x[1]

    # multi-channel input is averaged to mono, raw vectors are accepted
    st = Stft(hcat(x, x), 16000; winsize=512, winstep=256)
    @test get_spec(st) ≈ get_spec(Stft(x, 16000; winsize=512, winstep=256))
    @test Stft(Frames(audio; winsize=256, winstep=256)) isa Stft   # no overlap is fine
end

# ---------------------------------------------------------------------------- #
#                              element type flow                               #
# ---------------------------------------------------------------------------- #
@testset "Float32 stays Float32" begin
    for T in (Float32, Float64)
        audio  = Audio911.load(wav_file; format=T)
        frames = Frames(audio; winsize=512, winstep=256, type=hamming, preemph=0.97)
        @test eltype(get_window(frames)) == T
        @test eltype(get_data(frames)) == T
        @test eltype(get_energy(frames)) == T
        for fe in (Stft(frames), Cwt(frames; voices=4, freqrange=(100, 4000)))
            @test eltype(get_spec(fe)) == T
            @test eltype(get_freq(fe)) == T
            @test eltype(get_times(fe)) == T
            lin = LinSpec(fe; freqrange=(100, 4000), win_norm=true)
            mel = MelSpec(fe; nbands=20, freqrange=(100, 4000))
            brk = BarkSpec(fe; nbands=20, freqrange=(100, 4000))
            erb = ErbSpec(fe; nbands=20, freqrange=(100, 4000))
            for s in (lin, mel, brk, erb)
                @test eltype(get_spec(s)) == T
                @test eltype(get_freq(s)) == T
            end
            mfcc = Mfcc(mel; ncoeffs=13, lifter=22, energy=raw_energy)
            gtcc = Gtcc(erb; ncoeffs=13, rect=cubic_root)
            @test eltype(get_spec(mfcc)) == T
            @test eltype(get_spec(gtcc)) == T
            @test eltype(get_spec(Delta(Delta(mfcc)))) == T
            for D in (SpectralCentroid, SpectralCrest, SpectralDecrease, SpectralEntropy,
                      SpectralFlatness, SpectralFlux, SpectralKurtosis, SpectralRolloff,
                      SpectralSkewness, SpectralSlope, SpectralSpread, SpectralBandwidth)
                @test eltype(get_data(D(lin))) == T
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#                                cepstrum knobs                                #
# ---------------------------------------------------------------------------- #
@testset "cepstrum options" begin
    audio = Audio911.load(wav_file; format=Float64)
    stft  = Stft(audio; winsize=512, winstep=256, type=hamming)
    mel   = MelSpec(stft; nbands=26)
    n     = get_nframes(stft)

    m0 = Mfcc(mel; ncoeffs=13)
    @test size(get_spec(m0)) == (13, n)
    @test get_ncoeffs(m0) == 13
    @test get_parent(m0) === mel

    # the input spectrogram is never mutated by the floor
    copy_mel = copy(get_spec(mel))
    Mfcc(mel; ncoeffs=13, dither=true)
    @test get_spec(mel) == copy_mel

    # dct matrices
    D = dct_ortho(Float64, 8)
    @test D * D' ≈ I
    @test dct_htk(Float64, 8)[1, :] ≈ fill(sqrt(2 / 8), 8)
    @test dct_plain(Float64, 8)[1, :] ≈ ones(8)
    @test dct_htk(Float64, 8)[2:end, :] == D[2:end, :]

    # first / ncoeffs selection
    m1 = Mfcc(mel; ncoeffs=12, first=1)
    @test get_spec(m1) ≈ get_spec(Mfcc(mel; ncoeffs=13))[2:end, :]
    @test_throws ArgumentError Mfcc(mel; ncoeffs=20, first=10)

    # lifter
    L = 22
    ml = Mfcc(mel; ncoeffs=13, lifter=L)
    g  = [1 + L / 2 * sin(π * i / L) for i in 0:12]
    @test get_spec(ml) ≈ get_spec(m0) .* g
    ml1 = Mfcc(mel; ncoeffs=13, lifter=L, lifter_offset=1)
    g1  = [1 + L / 2 * sin(π * i / L) for i in 1:13]
    @test get_spec(ml1) ≈ get_spec(m0) .* g1

    # energy
    me = Mfcc(mel; ncoeffs=13, energy=raw_energy)
    @test get_spec(me)[1, :] ≈ log.(get_energy(stft))
    @test get_spec(me)[2:end, :] ≈ get_spec(m0)[2:end, :]
    ma = Mfcc(mel; ncoeffs=13, energy=spectrum_energy, energy_mode=:append)
    @test size(get_spec(ma), 1) == 14
    @test get_spec(ma)[14, :] ≈ log.(vec(sum(get_spec(stft), dims=1)))
    @test_throws ArgumentError Mfcc(mel; ncoeffs=12, first=1, energy=raw_energy)
    @test_throws ArgumentError Mfcc(mel; energy=raw_energy, energy_mode=:bogus)

    # rectifications and top_db clipping
    mn = Mfcc(mel; ncoeffs=13, rect=nlog)
    @test get_spec(mn) ≈ get_spec(m0) .* log(10)
    md = Mfcc(mel; ncoeffs=13, rect=db)
    @test get_spec(md) ≈ get_spec(m0) .* 10
    mt = Mfcc(mel; ncoeffs=13, rect=db, top_db=20)
    @test !(get_spec(mt) ≈ get_spec(md))
    @test nlog(ℯ) ≈ 1 && mlog(10) ≈ 1 && cubic_root(8) ≈ 2 && db(100) ≈ 20

    # delta from a raw matrix and options
    d = Delta(get_data(m0); sr=16000)
    @test get_data(d) ≈ get_data(Delta(m0))
    @test get_parent(d) === nothing
    @test_throws ArgumentError get_times(d)
    dt = Delta(m0; source=:transposed)
    @test size(get_data(dt)) == size(get_data(m0))
    @test_throws ArgumentError Delta(m0; source=:bogus)
end

# ---------------------------------------------------------------------------- #
#                           pad_end, scale, energy_floor                       #
# ---------------------------------------------------------------------------- #
@testset "pad_end, scale, energy_floor" begin
    audio = Audio911.load(wav_file; format=Float64)
    x = vec(get_data(audio)); n = length(x)
    fp = Frames(audio; winsize=400, winstep=160, pad_end=true)
    @test length(fp) == 1 + cld(n - 400, 160)
    @test length(fp) ≥ length(Frames(audio; winsize=400, winstep=160))
    last = get_data(fp)[:, end]
    @test last[1] == x[(length(fp) - 1) * 160 + 1]
    @test all(last[end - (length(fp) - 1) * 160 - 400 + n + 1:end] .== 0)
    short = Frames(x[1:100], 16000; winsize=400, winstep=160, pad_end=true)
    @test length(short) == 1 && get_data(short)[101:end, 1] == zeros(300)

    st  = Stft(audio; winsize=512, winstep=256)
    st2 = Stft(audio; winsize=512, winstep=256, scale=1 / 512)
    @test get_spec(st2) ≈ get_spec(st) ./ 512
    @test get_setup(st2).scale == 1 / 512

    mel = MelSpec(st; nbands=20)
    me = Mfcc(mel; ncoeffs=13, energy=raw_energy, energy_floor=1e3)
    @test all(get_spec(me)[1, :] .>= log(1e3))
end
