# ---------------------------------------------------------------------------- #
#                      non-negative matrix factorisation                       #
# ---------------------------------------------------------------------------- #
# Port of audioFlux's classic/nmf.c (MIT licence, Copyright (c) 2023
# libAudioFlux): multiplicative updates (Lee & Seung 2001; Févotte, Bertin &
# Durrieu 2009 for Itakura-Saito) with the basis columns renormalised after
# every iteration.

const _NMF_EPS = 1e-16

# NNDSVD-A initialisation (Boutsidis & Gallopoulos 2008; scikit-learn
# `nndsvda`): the leading singular triplets split into their positive and
# negative parts, zeros filled with the mean of V
function _nmf_nndsvd(V::AbstractMatrix{T}, k::Int) where T
    n, m = size(V)
    F = svd(Matrix(V))
    W = zeros(T, n, k); H = zeros(T, k, m)
    s1 = sqrt(F.S[1])
    W[:, 1] .= s1 .* abs.(view(F.U, :, 1))
    H[1, :] .= s1 .* abs.(view(F.Vt, 1, :))
    for j in 2:k
        x = view(F.U, :, j); y = view(F.Vt, j, :)
        xp, xn = max.(x, zero(T)), max.(.-x, zero(T))
        yp, yn = max.(y, zero(T)), max.(.-y, zero(T))
        np_, nn_ = norm(xp) * norm(yp), norm(xn) * norm(yn)
        u, v, σ = np_ ≥ nn_ ? (xp, yp, np_) : (xn, yn, nn_)
        σ > 0 || continue
        λ = sqrt(F.S[j] * σ)
        W[:, j] .= λ .* u ./ norm(u)
        H[j, :] .= λ .* v ./ norm(v)
    end
    avg = T(mean(V))
    W[W .< T(1e-6)] .= avg
    H[H .< T(1e-6)] .= avg
    return W, H
end

function _nmf_init(V::AbstractMatrix{T}, k::Int, init) where T
    n, m = size(V)
    if init === :nndsvd
        return _nmf_nndsvd(V, k)
    elseif init === :audioflux
        # audioFlux's Python wrapper: 1, 2, 3, ... in row-major order
        W = T[(i - 1) * k + c for i in 1:n, c in 1:k]
        H = T[(c - 1) * m + j for c in 1:k, j in 1:m]
        return W, H
    elseif init isa Tuple{AbstractMatrix,AbstractMatrix}
        W, H = Matrix{T}(init[1]), Matrix{T}(init[2])
        size(W) == (n, k) && size(H) == (k, m) ||
            throw(DimensionMismatch("init must be ($n × $k, $k × $m) matrices, got $(size(W)) and $(size(H))"))
        (all(≥(0), W) && all(≥(0), H)) || throw(ArgumentError("the initial W and H must be non-negative"))
        return W, H
    end
    throw(ArgumentError("init must be :nndsvd, :audioflux or a (W, H) tuple"))
end

# divide every column of W by its maximum, L1 or L2 norm (zeros stay zeros)
function _nmf_normalize!(W::AbstractMatrix{T}, norm::Symbol) where T
    @inbounds for c in axes(W, 2)
        col = view(W, :, c)
        d = norm === :max ? maximum(col) : norm === :l1 ? sum(abs, col) : sqrt(sum(abs2, col))
        d == 0 && continue
        col ./= d
    end
    return W
end

"""
    nmf(V, k; divergence=:kl, max_iter=300, thresh=1e-3, norm=:max, init=:nndsvd) -> (W, H)
    nmf(spec::AbstractSpectrogram, k; kwargs...) -> (W, H)

Non-negative matrix factorisation `V ≈ W H` of an `n × m` non-negative
matrix into `k` components (audioFlux `nmf`): on a spectrogram
(`bins × frames`), the columns of `W` are spectral templates and the rows
of `H` their activations over time.

Multiplicative updates minimise the generalised Kullback-Leibler
divergence (`divergence=:kl`, Lee & Seung), the Itakura-Saito divergence
(`:is`, Févotte et al.) or the Euclidean distance (`:euclidean`). As in
audioFlux, `H` is updated first, then `W`, both from the product `W H` of
the start of the iteration, and the columns of `W` are then divided by
their maximum (`norm=:max`), L1 (`:l1`) or L2 norm (`:l2`); the iterations
stop after `max_iter` or when both updates change `W` and `H` by less than
`thresh` (Frobenius norm).

`init` is `:nndsvd` (NNDSVD-A, deterministic), `:audioflux` (the ramps
`1, 2, 3, ...` in row-major order that audioFlux's Python wrapper starts
from) or a tuple `(W0, H0)`.

```julia
W, H = nmf(Stft(audio; spectrum=magnitude), 8)
```
"""
function nmf(V::AbstractMatrix{T}, k::Int; divergence::Symbol=:kl, max_iter::Int=300, thresh::Real=1e-3,
             norm::Symbol=:max, init=:nndsvd) where {T<:AbstractFloat}
    n, m = size(V)
    1 ≤ k ≤ min(n, m) || throw(ArgumentError("k must be in 1:$(min(n, m)), got $k"))
    divergence in (:kl, :is, :euclidean) || throw(ArgumentError("divergence must be :kl, :is or :euclidean, got :$divergence"))
    norm in (:max, :l1, :l2) || throw(ArgumentError("norm must be :max, :l1 or :l2, got :$norm"))
    all(≥(0), V) || throw(ArgumentError("V must be non-negative"))
    W, H = _nmf_init(V, k, init)
    e = T(_NMF_EPS)
    th = T(thresh)
    D = Matrix{T}(undef, n, m); R = similar(D)
    Q = divergence === :euclidean ? D : similar(D)
    divergence === :kl && fill!(Q, one(T))
    W0 = similar(W); H0 = similar(H)
    Hn = similar(H); Hd = similar(H); Wn = similar(W); Wd = similar(W)
    _nmf_normalize!(W, norm)
    for _ in 1:max_iter
        copyto!(W0, W); copyto!(H0, H)
        mul!(D, W, H)
        if divergence === :kl
            @. R = V / (D + e)
        elseif divergence === :is
            @. R = V / (D * D + e)
            @. Q = 1 / (D + e)
        end
        num = divergence === :euclidean ? V : R
        # 1. H ← H ⊙ Wᵀ num ⊘ Wᵀ Q
        mul!(Hn, W', num); mul!(Hd, W', Q)
        @. H = H * Hn / (Hd + e)
        # 2. W ← W ⊙ num Hᵀ ⊘ Q Hᵀ (with the same D)
        mul!(Wn, num, H'); mul!(Wd, Q, H')
        @. W = W * Wn / (Wd + e)
        _nmf_normalize!(W, norm)
        dw = sqrt(sum(abs2(W[i] - W0[i]) for i in eachindex(W)))
        dh = sqrt(sum(abs2(H[i] - H0[i]) for i in eachindex(H)))
        (dw < th && dh < th) && break
    end
    return W, H
end

nmf(s::AbstractSpectrogram, k::Int; kwargs...) = nmf(get_spec(s), k; kwargs...)
