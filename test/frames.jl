using Test
using Audio911

test_files_dir() = joinpath(dirname(@__FILE__), "test_files")
test_file(filename) = joinpath(test_files_dir(), filename)

wav_file = test_file("test.wav")
mp3_file = test_file("test.mp3")

audiofile = Audio911.load(wav_file; sr=8000, norm=false)

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

@btime Frames(audiofile);
# 3.493 μs (11 allocations: 10.41 KiB)
