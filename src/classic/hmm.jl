# ---------------------------------------------------------------------------- #
#                     hidden Markov models and Viterbi decoding                #
# ---------------------------------------------------------------------------- #
# Ports of audioFlux's classic/viterbi.c and classic/hmm.c (MIT licence,
# Copyright (c) 2023 libAudioFlux), with scaled forward-backward recursions
# (Rabiner 1989) and a backtracked Viterbi path.

"""
    viterbi(prob, A; p_init=nothing) -> (path, logp)

Most likely state sequence of a hidden Markov model (Viterbi 1967; librosa
`sequence.viterbi`, audioFlux `viterbi`). `prob` is `states × frames`, the
likelihood of every frame under every state (for a discrete HMM, the
emission column of each observed symbol); `A` is the `states × states`
transition matrix (rows sum to one, `A[i, j]` from `i` to `j`), and
`p_init` the initial distribution (uniform by default). The recursion runs
on log probabilities (`log(p + floatmin)`), the path is backtracked from the
best final state, and `logp` is its log probability.

audioFlux's `viterbi` returns the best state of every frame (the argmax of
the Viterbi scores) instead of the backtracked path; the score of the best
final state is the same.
"""
function viterbi(prob::AbstractMatrix{T}, A::AbstractMatrix; p_init=nothing) where {T<:Real}
    S, N = size(prob)
    size(A) == (S, S) || throw(DimensionMismatch("A must be $S × $S, got $(size(A))"))
    all(≥(0), prob) || throw(ArgumentError("prob must be non-negative"))
    _check_stochastic(A, "A")
    F = float(T)
    tiny = floatmin(F)
    π0 = isnothing(p_init) ? fill(F(1 / S), S) : F.(p_init)
    length(π0) == S || throw(DimensionMismatch("p_init must have $S states, got $(length(π0))"))
    _check_stochastic(reshape(π0, 1, S), "p_init")
    logA = log.(F.(A) .+ tiny)
    δ = Vector{F}(undef, S); δn = similar(δ)
    ptr = Matrix{Int}(undef, S, N)
    @inbounds for j in 1:S
        δ[j] = log(π0[j] + tiny) + log(F(prob[j, 1]) + tiny)
    end
    @inbounds for t in 2:N
        for j in 1:S
            best, bk = δ[1] + logA[1, j], 1
            for k in 2:S
                v = δ[k] + logA[k, j]
                v > best && ((best, bk) = (v, k))
            end
            δn[j] = best + log(F(prob[j, t]) + tiny)
            ptr[j, t] = bk
        end
        δ, δn = δn, δ
    end
    path = Vector{Int}(undef, N)
    N == 0 && return path, zero(F)
    logp, path[N] = findmax(δ)
    @inbounds for t in N:-1:2
        path[t - 1] = ptr[path[t], t]
    end
    return path, logp
end

function _check_stochastic(M::AbstractMatrix, name::String)
    all(≥(0), M) || throw(ArgumentError("$name must be non-negative"))
    all(r -> isapprox(sum(r), 1; atol=1e-4), eachrow(M)) ||
        throw(ArgumentError("every row of $name must sum to 1"))
    return nothing
end

"""
    Hmm{T}

A discrete hidden Markov model `λ = (π, A, B)` (audioFlux `hmm`): the
initial distribution `p_init` over `S` states, the `S × S` transition matrix
`A` (`A[i, j]`, from `i` to `j`) and the `S × K` emission matrix `B`
(`B[i, k]`, symbol `k` in state `i`); every row sums to one. States and
symbols are numbered from 1. See [`hmm_predict`](@ref),
[`hmm_decode`](@ref), [`hmm_train`](@ref) and [`hmm_generate`](@ref).

```julia
h = Hmm([0.6, 0.4], [0.7 0.3; 0.4 0.6], [0.5 0.4 0.1; 0.1 0.3 0.6])
states, obs = hmm_generate(h, 200)
h2 = hmm_train(Hmm(fill(0.5, 2), fill(0.5, 2, 2), [0.4 0.3 0.3; 0.2 0.3 0.5]), obs)
path, logp = hmm_decode(h2, obs)
```
"""
struct Hmm{T<:AbstractFloat}
    p_init :: Vector{T}
    A      :: Matrix{T}
    B      :: Matrix{T}
    function Hmm{T}(p_init::AbstractVector, A::AbstractMatrix, B::AbstractMatrix) where T
        S = length(p_init)
        size(A) == (S, S) || throw(DimensionMismatch("A must be $S × $S, got $(size(A))"))
        size(B, 1) == S || throw(DimensionMismatch("B must have $S rows, got $(size(B, 1))"))
        _check_stochastic(reshape(collect(p_init), 1, S), "p_init")
        _check_stochastic(A, "A"); _check_stochastic(B, "B")
        return new{T}(Vector{T}(p_init), Matrix{T}(A), Matrix{T}(B))
    end
end
Hmm(p_init::AbstractVector, A::AbstractMatrix, B::AbstractMatrix) =
    Hmm{float(promote_type(eltype(p_init), eltype(A), eltype(B)))}(p_init, A, B)
Base.show(io::IO, h::Hmm{T}) where T = print(io, "Hmm{$T}($(size(h.B, 1)) states, $(size(h.B, 2)) symbols)")

function _check_obs(h::Hmm, obs::AbstractVector{<:Integer})
    K = size(h.B, 2)
    isempty(obs) && throw(ArgumentError("the observation sequence is empty"))
    all(o -> 1 ≤ o ≤ K, obs) || throw(ArgumentError("observations must be symbols in 1:$K"))
    return nothing
end

# scaled forward (α̂, scale factors c with Π c = P(O | λ)) and backward (β̂) passes
function _forward(h::Hmm{T}, obs) where T
    S, N = size(h.A, 1), length(obs)
    α = Matrix{T}(undef, S, N); c = Vector{T}(undef, N)
    @inbounds for i in 1:S
        α[i, 1] = h.p_init[i] * h.B[i, obs[1]]
    end
    @inbounds for t in 1:N
        if t > 1
            for j in 1:S
                s = zero(T)
                for k in 1:S
                    s += α[k, t - 1] * h.A[k, j]
                end
                α[j, t] = s * h.B[j, obs[t]]
            end
        end
        c[t] = sum(view(α, :, t))
        c[t] > 0 && (view(α, :, t) ./= c[t])
    end
    return α, c
end

function _backward(h::Hmm{T}, obs, c) where T
    S, N = size(h.A, 1), length(obs)
    β = Matrix{T}(undef, S, N)
    β[:, N] .= one(T)
    @inbounds for t in N-1:-1:1
        for i in 1:S
            s = zero(T)
            for k in 1:S
                s += h.A[i, k] * h.B[k, obs[t + 1]] * β[k, t + 1]
            end
            β[i, t] = c[t + 1] > 0 ? s / c[t + 1] : s
        end
    end
    return β
end

"""
    hmm_predict(h::Hmm, obs) -> logp

Log-likelihood `log P(obs | λ)` of an observation sequence (symbols in
`1:K`) under the model, by the scaled forward algorithm (audioFlux
`hmmObj_predict`, which returns the unscaled probability and underflows on
long sequences).
"""
function hmm_predict(h::Hmm{T}, obs::AbstractVector{<:Integer}) where T
    _check_obs(h, obs)
    _, c = _forward(h, obs)
    return sum(log, c)
end

"""
    hmm_decode(h::Hmm, obs) -> (path, logp)

Most likely state sequence of the observations and its log probability,
[`viterbi`](@ref) on the emission columns of the observed symbols
(audioFlux `hmmObj_decode`).
"""
function hmm_decode(h::Hmm, obs::AbstractVector{<:Integer})
    _check_obs(h, obs)
    return viterbi(h.B[:, obs], h.A; p_init=h.p_init)
end

"""
    hmm_train(h::Hmm, obs; max_iter=100, tol=1e-3) -> Hmm

Baum-Welch re-estimation of the model from one observation sequence,
starting from `h` (audioFlux `hmmObj_train`): every iteration computes
the state and transition posteriors with the forward-backward algorithm
and re-estimates `π`, `A` and `B`; it stops after `max_iter` or when the
Euclidean norms of the changes of `π`, `A` and `B` are all at most `tol`.
Rows of states that are never visited keep their previous values.
audioFlux starts from a random model seeded by the clock; here the start is
the model given.
"""
function hmm_train(h::Hmm{T}, obs::AbstractVector{<:Integer}; max_iter::Int=100, tol::Real=1e-3) where T
    _check_obs(h, obs)
    S, K, N = size(h.A, 1), size(h.B, 2), length(obs)
    cur = h
    γ = Matrix{T}(undef, S, N)
    ξ = zeros(T, S, S); γs = zeros(T, S); Bn = zeros(T, S, K)
    for _ in 1:max_iter
        α, c = _forward(cur, obs)
        β = _backward(cur, obs, c)
        @inbounds for t in 1:N
            s = zero(T)
            for i in 1:S
                γ[i, t] = α[i, t] * β[i, t]; s += γ[i, t]
            end
            s > 0 && (view(γ, :, t) ./= s)
        end
        fill!(ξ, zero(T))
        @inbounds for t in 1:N-1
            s = zero(T)
            for i in 1:S, j in 1:S
                s += α[i, t] * cur.A[i, j] * cur.B[j, obs[t + 1]] * β[j, t + 1]
            end
            s > 0 || continue
            for i in 1:S, j in 1:S
                ξ[i, j] += α[i, t] * cur.A[i, j] * cur.B[j, obs[t + 1]] * β[j, t + 1] / s
            end
        end
        γs .= vec(sum(view(γ, :, 1:N-1); dims=2))
        fill!(Bn, zero(T))
        @inbounds for t in 1:N, i in 1:S
            Bn[i, obs[t]] += γ[i, t]
        end
        pn = γ[:, 1]
        An = copy(cur.A); Bnew = copy(cur.B)
        @inbounds for i in 1:S
            if γs[i] > 0
                An[i, :] .= view(ξ, i, :) ./ γs[i]
            end
            g = sum(view(γ, i, :))
            if g > 0
                Bnew[i, :] .= view(Bn, i, :) ./ g
            end
        end
        dp = sqrt(sum(abs2, pn .- cur.p_init))
        da = sqrt(sum(abs2, An .- cur.A))
        db = sqrt(sum(abs2, Bnew .- cur.B))
        cur = Hmm{T}(pn, An, Bnew)
        (dp ≤ tol && da ≤ tol && db ≤ tol) && break
    end
    return cur
end

# inverse-CDF draw of an index from the probabilities in `w`
function _draw(w::AbstractVector{T}, u::Real) where T
    acc = zero(T)
    @inbounds for i in eachindex(w)
        acc += w[i]
        u < acc && return i
    end
    return lastindex(w)
end

"""
    hmm_generate(h::Hmm, n; rng=nothing) -> (states, obs)

Sample a state sequence of length `n` and its observations from the model
(audioFlux `hmmObj_generate`). `rng` is any random number generator
(`Random.Xoshiro(1)`, ...), the global one by default.
"""
function hmm_generate(h::Hmm, n::Int; rng=nothing)
    n ≥ 0 || throw(ArgumentError("n must be ≥ 0"))
    u() = isnothing(rng) ? rand() : rand(rng)
    states = Vector{Int}(undef, n); obs = Vector{Int}(undef, n)
    n == 0 && return states, obs
    states[1] = _draw(h.p_init, u())
    for t in 2:n
        states[t] = _draw(view(h.A, states[t - 1], :), u())
    end
    for t in 1:n
        obs[t] = _draw(view(h.B, states[t], :), u())
    end
    return states, obs
end
