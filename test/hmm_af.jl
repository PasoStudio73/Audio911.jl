using Test
using Audio911
using MAT

af_files_dir()    = joinpath(dirname(@__FILE__), "audioflux_files", "hmm")
af_file(filename) = joinpath(af_files_dir(), filename)

# a small pseudo-random generator, so the tests do not depend on Random
mutable struct Lcg; s::UInt64; end
Base.rand(r::Lcg) = (r.s = r.s * 6364136223846793005 + 1442695040888963407; (r.s >> 11) / 2.0^53)

@testset "viterbi and Hmm" begin
    # a sticky two-state chain observed through noisy likelihoods
    A = [0.95 0.05; 0.05 0.95]
    truth = vcat(fill(1, 30), fill(2, 40), fill(1, 30))
    prob = [s == q ? 0.7 : 0.3 for q in 1:2, s in truth]
    prob[:, 10] = [0.2, 0.8]; prob[:, 50] = [0.8, 0.2]          # isolated outliers
    path, logp = viterbi(prob, A)
    @test path == truth
    @test logp ≈ log(0.5) + sum(log(prob[truth[t], t]) for t in 1:100) +
                 sum(log(A[truth[t-1], truth[t]]) for t in 2:100)
    # with a flat transition matrix it is the frame-wise argmax
    @test viterbi(prob, fill(0.5, 2, 2))[1] == [argmax(prob[:, t]) for t in 1:100]
    @test viterbi(prob, A; p_init=[0.0, 1.0])[1][1] == 2
    @test_throws DimensionMismatch viterbi(prob, fill(1 / 3, 3, 3))
    @test_throws ArgumentError viterbi(prob, [0.5 0.6; 0.5 0.5])
    @test_throws ArgumentError viterbi(-prob, A)

    h = Hmm([0.6, 0.4], [0.9 0.1; 0.2 0.8], [0.7 0.2 0.1; 0.1 0.3 0.6])
    @test sprint(show, h) == "Hmm{Float64}(2 states, 3 symbols)"
    states, obs = hmm_generate(h, 3000; rng=Lcg(1))
    @test length(states) == length(obs) == 3000
    @test all(in(1:2), states) && all(in(1:3), obs)
    # the empirical transition and emission frequencies match the model
    n12 = count(t -> states[t-1] == 1 && states[t] == 2, 2:3000) / count(==(1), states[1:end-1])
    @test isapprox(n12, 0.1; atol=0.03)
    @test isapprox(count(==(3), obs[states .== 2]) / count(==(2), states), 0.6; atol=0.05)
    @test hmm_generate(h, 0; rng=Lcg(1)) == (Int[], Int[])
    # predict: the forward likelihood equals the brute-force sum over paths
    o = [1, 3, 2, 3]
    brute = sum(Iterators.product(1:2, 1:2, 1:2, 1:2)) do q
        h.p_init[q[1]] * h.B[q[1], o[1]] * prod(h.A[q[t-1], q[t]] * h.B[q[t], o[t]] for t in 2:4)
    end
    @test hmm_predict(h, o) ≈ log(brute)
    @test isfinite(hmm_predict(h, obs))                 # no underflow on 3000 frames
    # decode recovers most of the hidden states
    path, _ = hmm_decode(h, obs)
    @test count(path .== states) / 3000 > 0.8
    # Baum-Welch increases the likelihood and approaches the true model
    h0 = Hmm([0.5, 0.5], [0.6 0.4; 0.4 0.6], [0.5 0.3 0.2; 0.2 0.3 0.5])
    h1 = hmm_train(h0, obs)
    @test hmm_predict(h1, obs) > hmm_predict(h0, obs)
    @test isapprox(h1.A, h.A; atol=0.08) && isapprox(h1.B, h.B; atol=0.08)
    @test all(isapprox.(sum(h1.A; dims=2), 1)) && all(isapprox.(sum(h1.B; dims=2), 1))
    @test hmm_train(h0, obs; max_iter=0) === h0
    @test_throws ArgumentError Hmm([0.5, 0.6], h.A, h.B)
    @test_throws DimensionMismatch Hmm([0.5, 0.5], fill(1 / 3, 3, 3), h.B)
    @test_throws ArgumentError hmm_predict(h, [1, 4])
    @test_throws ArgumentError hmm_decode(h, Int[])
    @test eltype(Hmm(Float32[0.5, 0.5], Float32[0.5 0.5; 0.5 0.5], Float32[1 0; 0 1]).A) == Float32
end

@testset "viterbi and Hmm against audioFlux" begin
    mat = MAT.matread(af_file("hmm.mat"))
    p, A, B = vec(mat["pi"]), mat["A"], mat["B"]
    obs = Int.(vec(mat["obs"]))
    # prob = viterbi(pi, A, B, S, K, obs, T, &isLog, sArr, mProbArr, mIndexArr)  (ctypes)
    for lg in (0, 1)
        path, logp = viterbi(B[:, obs], A; p_init=p)
        @test path == Int.(vec(mat["vit_path_$lg"]))        # audioFlux's pointers, backtracked
        ref = mat["vit_prob_$lg"]
        @test isapprox(logp, lg == 1 ? ref : log(ref); rtol=1e-5)
    end
    h = Hmm(p, A, B)
    # hmmObj_init(h, pi, A, B); hmmObj_predict(h, obs, T); hmmObj_decode(h, obs, T, sArr, prob)
    @test isapprox(hmm_predict(h, obs), log(mat["hmm_predict"]); rtol=1e-5)
    @test isapprox(hmm_decode(h, obs)[2], log(mat["hmm_decode"]); rtol=1e-5)
    # hmmObj_train(h, obs, T, &max_iter, &1e-3) from (train_pi0, train_A0, train_B0)
    h0 = Hmm(vec(mat["train_pi0"]), mat["train_A0"], mat["train_B0"])
    for iters in (1, 5, 100)
        ht = hmm_train(h0, obs; max_iter=iters, tol=1e-3)
        @test isapprox(ht.p_init, vec(mat["train_pi_$iters"]); atol=1e-4)
        @test isapprox(ht.A, mat["train_A_$iters"]; atol=1e-4)
        @test isapprox(ht.B, mat["train_B_$iters"]; atol=1e-4)
    end
end
