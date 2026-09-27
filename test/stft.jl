using Test
using Audio911

using InteractiveUtils

test_files_dir()    = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)

wav_file = test_file("test.wav")
mp3_file = test_file("test.mp3")

audiofile = Audio911.load(wav_file; sr=8000, norm=false)

@test_nowarn Stft(audiofile)

@test_nowarn Stft(audiofile; winsize=1024, winstep=512)
@test_nowarn Stft(audiofile; winsize=512, winstep=256)

frames = Frames(audiofile; winsize=512, winstep=256, type=hamming, periodic=true)

@test_nowarn Stft(frames)

@test_nowarn Stft(frames; nfft=1024)
@test_nowarn Stft(frames; nfft=2048)
@test_throws ArgumentError Stft(frames; nfft=128)

@test_nowarn Stft(frames; spectrum=power)
@test_nowarn Stft(frames; spectrum=magnitude)

stft = Stft(frames; spectrum=power)

# ---------------------------------------------------------------------------------------- #
#                                      code warntype                                       #
# ---------------------------------------------------------------------------------------- #
section(s) = println("\n", "="^90, "\n  ", s, "\n", "="^90)

audiofile = Audio911.load(wav_file; sr=8000)
frames = Frames(audiofile; winsize=512, winstep=256, type=hamming)
stft = Stft(frames)
stft_c = Stft(frames; keep_complex=true)
stft_m = Stft(frames; spectrum=magnitude, scale=1/512)

# normalizations
section("power / magnitude")
z = rand(Complex{Float32}, 257, 10)
@code_warntype Audio911.power(z)
@code_warntype Audio911.power(z[1])
@code_warntype Audio911.magnitude(z)
@code_warntype Audio911.magnitude(z[1])

# utilities
section("utilities")
@code_warntype Audio911._onesided_length(512)
@code_warntype Audio911._chunks(100)

# _stft!
section("_stft!")
spec = Matrix{Float32}(undef, 257, length(frames))
cspec = Matrix{Complex{Float32}}(undef, 257, length(frames))
w = get_window(frames)
@code_warntype Audio911._stft!(spec, frames, 512, Audio911.power)
@code_warntype Audio911._stft!(spec, frames, 1024, Audio911.magnitude, w)
@code_warntype Audio911._stft!(cspec, frames, 512, identity)

# constructors
section("Stft constructors")
@code_warntype Stft(frames)
@code_warntype Stft(frames; nfft=1024, spectrum=magnitude, scale=0.5)
@code_warntype Stft(frames; keep_complex=true)
@code_warntype Stft(get_data(audiofile), get_sr(audiofile))
@code_warntype Stft(get_data(audiofile), 8000; winsize=1024, winstep=512,
    type=povey, periodic=false, center=true, preemph=0.97)
@code_warntype Stft(audiofile)

# accessors
section("accessors")
@code_warntype Audio911.get_data(stft)
@code_warntype Audio911.get_spec(stft)
@code_warntype Audio911.get_freq(stft)
@code_warntype Audio911.get_setup(stft)
@code_warntype Audio911.get_sr(stft)
@code_warntype Audio911.get_nfft(stft)
@code_warntype Audio911.get_spectrum(stft)
@code_warntype Audio911.get_winsize(stft)
@code_warntype Audio911.get_step(stft)
@code_warntype Audio911.get_overlap(stft)
@code_warntype Audio911.get_offset(stft)
@code_warntype Audio911.get_window(stft)
@code_warntype Audio911.get_frames(stft)
@code_warntype Audio911.get_parent(stft)
@code_warntype Audio911.get_energy(stft)
@code_warntype Audio911.get_complex(stft) # recompute branch
@code_warntype Audio911.get_complex(stft_c) # cached branch
@code_warntype Audio911._freq_indices(stft, FreqRange(300, 3400))

# @inferred summary
section("@inferred summary")
@testset "stft type stability" begin
    @test @inferred(Audio911.power(z)) isa Matrix{Float32}
    @test @inferred(Audio911.magnitude(z)) isa Matrix{Float32}
    @test @inferred(Audio911._onesided_length(512)) == 257
    @test @inferred(Audio911._stft!(spec, frames, 512, Audio911.power)) isa Matrix{Float32}
    @test @inferred(Audio911._stft!(cspec, frames, 512, identity)) isa Matrix{Complex{Float32}}
    @test @inferred(Stft(frames)) isa Stft{Float32}
    @test @inferred(Audio911.get_data(stft)) isa Matrix{Float32}
    @test @inferred(Audio911.get_freq(stft)) isa StepRangeLen{Float32}
    @test @inferred(Audio911.get_energy(stft)) isa Vector{Float32}
    @test @inferred(Audio911.get_complex(stft)) isa Matrix{Complex{Float32}}
    @test @inferred(Audio911.get_complex(stft_c)) isa Matrix{Complex{Float32}}
end

@btime Stft(audiofile);
# 69.588 μs (30 allocations: 76.24 KiB)
