using Test
using Audio911

# allocation bounds for the hot paths: a small multiple of the output size on
# a 60 s Float32 signal at 44.1 kHz. Each stage is called once to compile.
sr = 44100
x  = 0.5f0 .* (2 .* rand(Float32, 60sr) .- 1)
MiB = 2^20

frames = Frames(x, sr; winsize=512, winstep=256, type=hamming)
n      = length(frames)
out_stft = 257 * n * sizeof(Float32)

@testset "allocation regression" begin
    Frames(x, sr; winsize=512, winstep=256, type=hamming)
    @test (@allocated Frames(x, sr; winsize=512, winstep=256, type=hamming)) < 64 * 1024

    stft = Stft(frames)
    @test (@allocated Stft(frames)) < 2 * out_stft

    mel = MelSpec(stft; nbands=26)
    @test (@allocated MelSpec(stft; nbands=26)) < 3 * 26 * n * sizeof(Float32) + 2MiB

    lin = LinSpec(stft; freqrange=(100, 4000))
    @test (@allocated LinSpec(stft; freqrange=(100, 4000))) < 2 * size(get_spec(lin), 1) * n * sizeof(Float32) + MiB

    mfcc = Mfcc(mel; ncoeffs=13)
    @test (@allocated Mfcc(mel; ncoeffs=13)) < 4 * 26 * n * sizeof(Float32) + MiB

    Delta(mfcc)
    @test (@allocated Delta(mfcc)) < 3 * 13 * n * sizeof(Float32) + MiB

    for D in (SpectralCentroid, SpectralSpread, SpectralFlux, SpectralRolloff, SpectralEntropy, SpectralFlatness)
        D(lin)
        @test (@allocated D(lin)) < 4 * n * sizeof(Float32) + 256 * 1024
    end

    # the input spectrogram survives untouched
    S = copy(get_spec(mel))
    Mfcc(mel; ncoeffs=13, dither=true)
    @test get_spec(mel) == S
end
