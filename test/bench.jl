# Pipeline benchmark: time, allocations and peak RSS of the main stages.
#
#   test/run.sh bench                      # this tree
#   julia --project=<other tree> test/bench.jl   # any tree exposing the same API
#
# Only API shared by the pre-refactor package and the current one is used, so
# the same script produces the "before" and "after" columns of
# docs/src/performance.md. Each stage runs twice; the second run is reported.
using Audio911
using Printf

const HERE = @__DIR__
wav = joinpath(HERE, "test_files", "test.wav")

struct Row
    name::String
    time::Float64
    bytes::Int
end
rows = Row[]

function bench!(name, f)
    f()                                  # warm-up (compilation)
    GC.gc()
    r = @timed f()
    push!(rows, Row(name, r.time, r.bytes))
    return r.value
end

function pipeline(audio, tag; sr=nothing)
    mk = isnothing(sr) ? (() -> Stft(audio; winsize=512, winstep=256, type=hamming, periodic=true)) :
                         (() -> Stft(audio, sr; winsize=512, winstep=256, type=hamming, periodic=true))
    stft = bench!("$tag Stft (512/256 hamming)", mk)
    mel  = bench!("$tag MelSpec 26 bands", () -> MelSpec(stft; nbands=26))
    mfcc = bench!("$tag Mfcc 13", () -> Mfcc(mel; ncoeffs=13))
    bench!("$tag Delta", () -> Delta(mfcc))
    erb  = bench!("$tag ErbSpec 26 bands", () -> ErbSpec(stft; nbands=26))
    bench!("$tag Gtcc 13", () -> Gtcc(erb; ncoeffs=13))
    lin  = bench!("$tag LinSpec", () -> LinSpec(stft; freqrange=(100, 4000)))
    bench!("$tag 11 spectral descriptors", () -> (SpectralCentroid(lin), SpectralCrest(lin), SpectralDecrease(lin),
        SpectralEntropy(lin), SpectralFlatness(lin), SpectralFlux(lin), SpectralKurtosis(lin), SpectralRolloff(lin),
        SpectralSkewness(lin), SpectralSlope(lin), SpectralSpread(lin)))
end

audio = bench!("load test.wav (Float32)", () -> load(wav))
pipeline(audio, "wav")

# 60 s of noise at 44.1 kHz, the memory-relevant case
sr = 44100
x32 = 0.5f0 .* (2 .* rand(Float32, 60sr) .- 1)
x64 = Float64.(x32)
pipeline(x32, "60s@44.1k Float32"; sr)
pipeline(x64, "60s@44.1k Float64"; sr)

println("| stage | time (s) | allocated (MiB) |")
println("|:------|---------:|----------------:|")
for r in rows
    @printf("| %s | %.3f | %.1f |\n", r.name, r.time, r.bytes / 2^20)
end
@printf("\npeak RSS of this process: %.0f MiB\n", Sys.maxrss() / 2^20)
