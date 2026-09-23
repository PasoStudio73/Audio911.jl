using Test
using Audio911

using MAT

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)
invalid_file(filename) = joinpath(test_files_dir(), "invalid", filename)

wav_file  = test_file("test.wav")
mp3_file  = test_file("test.mp3")
flac_file = test_file("test.flac")
ogg_file  = test_file("test.ogg")

# ---------------------------------------------------------------------------- #
#                                audio reader                                  #
# ---------------------------------------------------------------------------- #
@testset "audioreader" begin
    @test_nowarn Audio911.File{format"WAV"}(wav_file)
    @test_nowarn Audio911.File{format"MP3"}(mp3_file)
    @test formatname(Audio911.File{format"WAV"}(wav_file)) == :WAV
    @test file_extension(Audio911.File{format"MP3"}(mp3_file)) == ".mp3"
    @test Audio911.detect_format(wav_file) == :WAV
    @test Audio911.detect_format(mp3_file) == :MP3
    @test Audio911.detect_format(flac_file) == :FLAC
    @test Audio911.detect_format(ogg_file) == :OGG

    @test_nowarn Audio911.load(wav_file)
    @test_nowarn Audio911.load(mp3_file)
    @test_nowarn Audio911.load(flac_file)
    @test_nowarn Audio911.load(ogg_file)

    audiofile = Audio911.load(wav_file; format=Float64)
    @test Audio911.get_data(audiofile) isa Array{Float64}
    @test Audio911.get_sr(audiofile) == 16000
    @test Audio911.get_origin_sr(audiofile) == 16000
    @test Audio911.get_nchannels(audiofile) == 1
    @test Audio911.is_norm(audiofile) == false
    @test get_path(audiofile) == wav_file
    @test get_duration(audiofile) ≈ length(audiofile) / 16000

    audiofile = Audio911.load(mp3_file; norm=true)
    @test Audio911.get_data(audiofile) isa Array{Float32}
    @test Audio911.get_sr(audiofile) == 44100
    @test Audio911.get_origin_sr(audiofile) == 44100
    @test Audio911.get_nchannels(audiofile) == 1
    @test Audio911.is_norm(audiofile) == true
    @test maximum(abs, get_data(audiofile)) ≈ 1

    audiofile = Audio911.load(mp3_file; mono=false)
    @test Audio911.get_data(audiofile) isa Array{Float32}
    @test Audio911.get_sr(audiofile) == 44100
    @test Audio911.get_origin_sr(audiofile) == 44100
    @test Audio911.get_nchannels(audiofile) == 2
    @test Audio911.is_norm(audiofile) == false

    audiofile = Audio911.load(wav_file; sr=48000)
    @test Audio911.get_data(audiofile) isa Array{Float32}
    @test Audio911.get_sr(audiofile) == 48000
    @test Audio911.get_origin_sr(audiofile) == 16000
    @test Audio911.get_nchannels(audiofile) == 1
    @test Audio911.is_norm(audiofile) == false

    audiofile = Audio911.load(mp3_file; mono=false, sr=8000)
    @test Audio911.get_nchannels(audiofile) == 2
    @test eltype(audiofile) == Float32
    @test length(audiofile) == 44513

    # lossless and lossy containers of the same 44.1 kHz recording
    flac = Audio911.load(flac_file; format=Float64, mono=false)
    ogg  = Audio911.load(ogg_file; mono=false)
    @test get_sr(flac) == 44100 && get_sr(ogg) == 44100
    @test get_nchannels(flac) == get_nchannels(ogg)
    @test eltype(flac) == Float64 && eltype(ogg) == Float32
    @test length(ogg) > 44100 && length(flac) > 44100
    @test all(isfinite, get_data(ogg))
    @test maximum(abs, get_data(flac)) <= 1
    @test_throws ArgumentError Audio911.load(wav_file; format=Int16)
end

# ---------------------------------------------------------------------------- #
#                                 invalid files                                #
# ---------------------------------------------------------------------------- #
@testset "invalid files" begin
    @test_throws ArgumentError Audio911.load(invalid_file("text.txt"))          # unsupported extension
    @test_throws ArgumentError Audio911.load(invalid_file("test.oga"))          # unsupported extension
    @test_throws ArgumentError Audio911.load(invalid_file("text.wav"))          # not a RIFF/WAVE file
    @test_throws ArgumentError Audio911.load(invalid_file("test_is_a_wav.mp3")) # content does not match extension
    @test_throws ArgumentError Audio911.load(test_file("missing.wav"))          # does not exist
end

# ---------------------------------------------------------------------------- #
#                              in-memory audio                                 #
# ---------------------------------------------------------------------------- #
@testset "in-memory AudioFile" begin
    x = rand(Float32, 8000, 2) .- 0.5f0
    a = AudioFile(x, 8000)
    @test get_nchannels(a) == 1 && length(a) == 8000 && get_sr(a) == 8000
    @test get_data(a) ≈ sum(x, dims=2) ./ 2
    @test get_path(a) == ""
    b = AudioFile(x, 8000; mono=false, norm=true, new_sr=4000, format=Float64)
    @test get_nchannels(b) == 2 && get_sr(b) == 4000 && get_origin_sr(b) == 8000
    @test eltype(b) == Float64 && is_norm(b)
    @test maximum(abs, get_data(b)) ≈ 1
    c = AudioFile(vec(x[:, 1]), 8000)
    @test size(get_data(c)) == (8000, 1)
    d = AudioFile(rand(Int16, 100), 8000)
    @test eltype(d) == Float32
    @test_throws ArgumentError AudioFile(x, 0)
    # signal utilities
    @test to_mono(x) ≈ sum(x, dims=2) ./ 2
    @test normalize_peak([0.5, -0.25]) == [1.0, -0.5]
    @test normalize_peak(zeros(3)) == zeros(3)
    @test size(Audio911.resample(x, 8000, 16000), 1) == 16000
    @test Audio911.resample(x, 8000, 8000) === x
    # the whole pipeline accepts in-memory audio
    @test Stft(a) isa Stft
    @test Frames(get_data(a), 8000) isa Frames
end

# ---------------------------------------------------------------------------- #
#                            test against matlab                               #
# ---------------------------------------------------------------------------- #
matlab_files_dir()    = joinpath(dirname(@__FILE__), "matlab_files/audioread")
matlab_file(filename) = joinpath(matlab_files_dir(), filename)

@testset "against matlab" begin
    matfile_wav = matlab_file("matlab_audioread_wav.mat")
    matfile_mp3 = matlab_file("matlab_audioread_mp3.mat")

    mat_wav = MAT.matread(matfile_wav)
    wav_data_mat = mat_wav["audio_wav"]

    a911_wav = Audio911.load(wav_file, format=Float64)
    wav_data_a911 = Audio911.get_data(a911_wav)

    @test isapprox(wav_data_a911, wav_data_mat)

    mat_mp3 = MAT.matread(matfile_mp3)
    mp3_data_mat = mat_mp3["audio_mp3"]

    a911_mp3 = Audio911.load(mp3_file, mono=false)
    mp3_data_a911 = Audio911.get_data(a911_mp3)

    @test isapprox(mp3_data_mat, mp3_data_a911)
end

# ---------------------------------------------------------------------------- #
#                                  resampling                                  #
# ---------------------------------------------------------------------------- #
@testset "resampling" begin
    orig_file = Audio911.load(wav_file)
    res_file  = Audio911.load(wav_file, sr=8000)
    @test length(res_file) == length(orig_file) ÷ 2
    @test get_sr(res_file) == 8000
    # a resampled tone keeps its frequency
    sr, f0 = 16000, 440.0
    t = (0:sr-1) ./ sr
    tone = AudioFile(sin.(2π * f0 .* t), sr; new_sr=48000, format=Float64)
    spec = Stft(tone; winsize=4096, winstep=2048)
    peak = get_freq(spec)[argmax(vec(sum(get_spec(spec), dims=2)))]
    @test isapprox(peak, f0; atol=get_freq(spec)[2])
end
