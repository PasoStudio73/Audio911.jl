using Test
using Audio911

test_files_dir()    = joinpath(dirname(@__FILE__), "wav_sample_files")
test_file(filename) = joinpath(test_files_dir(), filename)

filenames = filter(name -> isfile(joinpath(test_files_dir(), name)),
    readdir(test_files_dir()))

for filename in filenames
    f = test_file(filename)
    sym = detect_format(f)
    @test sym == :WAV
    audiofile = Audio911.load(f; format=Float32, mono=false)
    @test get_sr(audiofile) == 8000
    @test is_norm(audiofile) == false
    @test size(get_data(audiofile), 2) == 2
end

