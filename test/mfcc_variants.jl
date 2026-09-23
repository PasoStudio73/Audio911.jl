using Test
using Audio911

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
wav_file = test_file("test.wav")

audio = Audio911.load(wav_file; format=Float64)
sr    = get_sr(audio)
x     = vec(get_data(audio))
N     = length(x)
nfr(ws, st) = fld(N - ws, st) + 1

@testset "mfcc variants" begin
    # MATLAB defaults: 30 ms window, 20 ms overlap, 13 coefficients
    m = mfcc_matlab(audio)
    @test size(get_data(m)) == (nfr(480, 160), 13)
    @test get_setup(m).rect === mlog
    @test get_nbands(get_parent(m)) == 32
    @test get_spectrum(m) === power
    # the preset is the same chain the fixtures test
    frames = Frames(audio; winsize=512, winstep=256, type=hamming, periodic=true)
    ref = Mfcc(MelSpec(Stft(frames); win_norm=true, nbands=32); ncoeffs=13, rect=mlog)
    @test get_data(mfcc_matlab(audio; winsize=512, winstep=256)) ≈ get_data(ref)

    # HTK: c1..c12, magnitude spectrum, warped unnormalised filterbank, lifter 22
    h = mfcc_htk(audio)
    @test size(get_data(h)) == (nfr(400, 160), 12)
    @test get_setup(h).first == 1 && get_setup(h).lifter == 22 && get_setup(h).dct === dct_htk
    @test get_spectrum(h) === magnitude
    @test get_setup(get_frames(h)).preemph == 0.97
    @test get_norm(get_fbank(get_parent(h))) === none_norm
    he = mfcc_htk(audio; energy=true, c0=true)
    @test size(get_data(he), 2) == 14            # c0..c12 + energy
    @test get_data(he)[:, 14] ≈ log.(get_energy(get_frames(he)))
    @test get_data(he)[:, 2:13] ≈ get_data(h)    # c1..c12 unchanged
    # scaling the input only shifts c0 / energy, never c1..c12
    h2 = mfcc_htk(audio; scale=1)
    @test !(get_data(h2) ≈ get_data(h))          # the floor bites on unscaled audio

    # Kaldi: 13 coefficients with C0 = log raw energy, povey window, dc removal
    k = mfcc_kaldi(audio)
    @test size(get_data(k)) == (nfr(400, 160), 13)
    fk = get_frames(k)
    @test get_setup(fk).type === povey && get_setup(fk).dc_removal && !get_setup(fk).periodic
    @test get_nfft(get_frontend(k)) == 512
    @test get_data(k)[:, 1] ≈ log.(max.(get_energy(fk), eps(Float32)))
    @test get_freqrange(get_parent(k)) == (20, 8000)
    k0 = mfcc_kaldi(audio; energy=false)
    @test get_data(k0)[:, 2:end] ≈ get_data(k)[:, 2:end]
    @test !(get_data(k0)[:, 1] ≈ get_data(k)[:, 1])

    # librosa: centred frames -> 1 + N ÷ hop frames, 20 coefficients, slaney mel
    l = mfcc_librosa(audio)
    @test size(get_data(l)) == (1 + fld(N, 512), 20)
    @test get_scale(get_fbank(get_parent(l))) == :slaney
    @test get_nbands(get_parent(l)) == 128
    @test get_setup(l).rect === db && get_setup(l).top_db == 80 && get_setup(l).lifter_offset == 1
    @test get_times(l)[1] ≈ 0
    l2 = mfcc_librosa(audio; nfft=1024, winstep=256, nbands=40, ncoeffs=13, top_db=nothing)
    @test size(get_data(l2)) == (1 + fld(N, 256), 13)

    # ETSI: 13 cepstra + log energy, integer-bin filterbank, plain DCT
    e = mfcc_etsi(audio)
    @test size(get_data(e)) == (nfr(400, 160), 14)
    @test get_setup(e).dct === dct_plain
    @test all(get_data(e)[:, 14] .>= -50)
    fb = etsi_fbank(16000; nfft=512)
    @test size(get_data(fb)) == (23, 257)
    @test all(0 .<= get_data(fb) .<= 1)
    @test all(sum(get_data(fb), dims=2) .> 0)
    @test get_freq(fb)[1] > 64

    # python_speech_features: padded last frame, rect window, C0 = log(Σ spectrum)
    p = mfcc_psf(audio)
    npsf = 1 + ceil(Int, (N - 400) / 160)
    @test size(get_data(p)) == (npsf, 13)
    @test get_setup(get_frames(p)).type === rect
    st = get_frontend(p)
    @test get_data(p)[:, 1] ≈ log.(max.(vec(sum(get_spec(st), dims=1)), eps(Float64)))
    fbp = psf_fbank(16000; nfft=512)
    @test size(get_data(fbp)) == (26, 257)
    @test maximum(get_data(fbp)) ≈ 1

    # offset compensation removes DC
    y = offset_compensation(fill(0.5, 12000))
    @test abs(y[end]) < 1e-3

    # every preset runs in Float32 and stays Float32
    a32 = Audio911.load(wav_file; format=Float32)
    for f in (mfcc_matlab, mfcc_htk, mfcc_kaldi, mfcc_librosa, mfcc_etsi, mfcc_psf)
        r = f(a32)
        @test eltype(get_spec(r)) == Float32
        @test all(isfinite, get_spec(r))
    end
end
