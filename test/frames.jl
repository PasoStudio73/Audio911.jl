using Test
using Audio911

using InteractiveUtils

test_files_dir() = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)

wav_file = test_file("test.wav")
mp3_file = test_file("test.mp3")

audiofile = Audio911.load(wav_file; sr=8000, norm=false)

# ---------------------------------------------------------------------------------------- #
#                                        frames                                            #
# ---------------------------------------------------------------------------------------- #
@test_nowarn Frames(audiofile)
@test_nowarn Frames(audiofile; winsize=256, winstep=128, type=hamming, periodic=true)

frames_data = Frames(audiofile; winsize=256, winstep=128, type=hamming)
@test length(frames_data) == 132

audiofile = load(wav_file; sr=6000)
frames = Frames(audiofile)
@test_nowarn get_data(frames)
@test get_size(frames) == 256
@test get_step(frames) == 128
@test get_overlap(frames) == 128

audiofile = load(wav_file; sr=10000)
frames = Frames(audiofile)
@test_nowarn get_data(frames)
@test get_size(frames) == 512
@test get_step(frames) == 256
@test get_overlap(frames) == 256

audiofile = Audio911.load(mp3_file)
frames = Frames(audiofile)
@test_nowarn get_data(frames)

# the `win` keyword accepts movingwindow(...) for compatibility
a1 = load(mp3_file)
f1 = Frames(a1; win=movingwindow(winsize=512, winstep=256))
f1 = Frames(a1; winsize=512, winstep=256)
@test get_data(f1) == get_data(f1)
@test get_winsize(Stft(a1; win=movingwindow(winsize=1024))) == 1024

# ---------------------------------------------------------------------------------------- #
#                                      code warntype                                       #
# ---------------------------------------------------------------------------------------- #
wav_file  = test_file("test.wav")
mp3_file  = test_file("test.mp3")
flac_file = test_file("test.flac")
ogg_file  = test_file("test.ogg")

v32 = rand(Float32, 8000)
af = AudioFile(v32, 8000)

section(s) = println("\n", "="^90, "\n  ", s, "\n", "="^90)

# frames.jl
section("frames.jl — windows")
@code_warntype Audio911.povey(256)
@code_warntype Audio911.bohman(256)
@code_warntype Audio911.kaiser(256; β=5)
@code_warntype Audio911.gauss(256; α=2.5)
@code_warntype Audio911.point(256)
@code_warntype Audio911.triangular(256)
@code_warntype Audio911.etsi(256)
@code_warntype Audio911._make_window(Float32, hanning, 256, true)
@code_warntype Audio911.movingwindow(winsize=512, winstep=256)
@code_warntype Audio911._winparams(nothing, 512, 256)
@code_warntype Audio911._winparams(movingwindow(winsize=512), 256, 128)

section("frames.jl — pre-emphasis / padding")
@code_warntype Audio911.preemphasis(v32; coef=0.97)
@code_warntype Audio911.deemphasis(v32; coef=0.97)
@code_warntype Audio911._pad_center(v32, 128, :constant)
@code_warntype Audio911._pad_center(v32, 128, :reflect)

section("frames.jl — Frames")
frames = Frames(v32, 8000)
@code_warntype Frames(v32, 8000)
@code_warntype Frames(v32, 8000; winsize=400, winstep=160, type=povey,
    periodic=false, preemph=0.97, dc_removal=true, center=true)
@code_warntype Frames(af)
@code_warntype Audio911.get_size(frames)
@code_warntype Audio911.get_step(frames)
@code_warntype Audio911.get_overlap(frames)
@code_warntype Audio911.get_window(frames)
@code_warntype Audio911.get_signal(frames)
@code_warntype Audio911.get_energy(frames)
buf = view(Matrix{Float32}(undef, 256, 1), :, 1)
@code_warntype Audio911.frame!(buf, frames, 1)
@code_warntype Audio911.get_data(frames)
@code_warntype Audio911.get_winframes(frames)

# automated pass
section("@inferred summary")
@testset "type stability" begin
    @test @inferred(Audio911.get_data(af)) isa Vector{Float32}
    @test @inferred(Audio911.get_duration(af)) isa Float64
    @test @inferred(Audio911.preemphasis(v32)) isa Vector{Float32}
    @test @inferred(Audio911.deemphasis(v32)) isa Vector{Float32}
    @test @inferred(Audio911._pad_center(v32, 128, :constant)) isa Vector{Float32}
    @test @inferred(Audio911._make_window(Float32, hanning, 256, true)) isa Vector{Float32}
    @test @inferred(Audio911.get_data(frames)) isa Matrix{Float32}
    @test @inferred(Audio911.get_energy(frames)) isa Vector{Float32}
    @test @inferred(Audio911.frame!(buf, frames, 1)) isa AbstractVector{Float32}
end

@btime Frames(audiofile);
# 3.493 μs (11 allocations: 10.41 KiB)
