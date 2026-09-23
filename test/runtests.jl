using Test

function run_tests(list)
    println("\n" * ("#"^50))
    for test in list
        println("TEST: $test")
        include(test)
    end
end

println("Julia version: ", VERSION)

test_suites = [
    ("Audioreader", ["audioreader.jl",]),
    ("Frames", ["frames.jl",]),
    ("STFT", ["stft.jl",]),
    ("FilterBanks", ["fbank.jl",]),
    ("Lin Spec", ["lin_spec.jl",]),
    ("Mel Spec", ["mel_spec.jl",]),
    ("Bark Spec", ["bark_spec.jl",]),
    ("Erb Spec", ["erb_spec.jl",]),
    ("Mfcc", ["mfcc.jl",]),
    ("Gtcc", ["gtcc.jl",]),
    ("Spectral", ["spectral.jl"]),
    ("Pipeline", ["pipeline.jl"]),
    ("Mfcc variants", ["mfcc_variants.jl"]),
    ("Features", ["features.jl"]),
    ("Signal utils", ["signal.jl"]),
    ("Plots", ["plots.jl"]),
    ("Allocations", ["allocations.jl"]),
    ("audioFlux filterbanks", ["af_fbank.jl"]),
    ("CQT", ["cqt.jl"]),
    ("Transforms", ["transforms.jl"]),
    ("Complex STFT and reassignment", ["reassign.jl"]),
    ("Wavelets and synchrosqueezing", ["cwt_af.jl"]),
    ("Discrete wavelets", ["dwt.jl"]),
    ("Cohen class and adaptive decompositions", ["adaptive.jl"]),
    ("audioFlux spectral descriptors", ["spectral_af.jl"]),
    ("audioFlux features", ["af_features.jl"]),
    ("Pitch estimators", ["pitch_af.jl"]),
    ("Novelty onsets", ["onset_af.jl"]),
    ("HPSS signals", ["hpss_af.jl"]),
    ("NMF", ["nmf_af.jl"]),
    ("HMM and Viterbi", ["hmm_af.jl"]),
    ("Harmonic count", ["harmonic_af.jl"]),
]

@testset "Audio911.jl" begin
    for ts in eachindex(test_suites)
        name = test_suites[ts][1]
        list = test_suites[ts][2]
        let
            @testset "$name" begin
                run_tests(list)
            end
        end
    end
    println()
end
