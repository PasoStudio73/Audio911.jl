# ---------------------------------------------------------------------------- #
#                                    info                                      #
# ---------------------------------------------------------------------------- #
struct DeltaSetup <: AbstractSetup
    sr           :: Int64
    delta_length :: Int64
    source       :: Symbol
end

# ---------------------------------------------------------------------------- #
#                                 delta struct                                 #
# ---------------------------------------------------------------------------- #
"""
    Delta{F,T} <: AbstractDelta

Temporal derivative of a feature matrix (MATLAB's `audioDelta`, `mfccDelta`),
stored as `coeffs × frames`. Apply it twice for delta-delta.

See [`Delta(x; delta_length)`](@ref Delta(::AbstractAudioSpectrum)).
"""
struct Delta{F,T<:AudioData} <: AbstractDelta
    spec   :: Matrix{T}
    parent :: F
    info   :: DeltaSetup
end

# causal FIR `y[:, j] = Σ b[i+1] x[:, j-i]` along the frame axis, zero initial state
function _delta(x::AbstractMatrix{T}, delta_length::Int64) where T
    m = delta_length ÷ 2
    den = T(sum((1:m) .^ 2))
    b = T[T(m - i) / den for i in 0:2m]
    nr, nc = size(x)
    y = zeros(T, nr, nc)
    @inbounds for j in 1:nc, i in 0:min(2m, j - 1)
        bi = b[i + 1]
        iszero(bi) && continue
        @simd for r in 1:nr
            y[r, j] += bi * x[r, j - i]
        end
    end
    return y
end

function _delta_matrix(x::AbstractMatrix{T}, delta_length::Int64, source::Symbol) where T
    delta_length > 2 || throw(ArgumentError("delta_length must be > 2, got $delta_length."))
    isodd(delta_length) || throw(ArgumentError("delta_length must be odd, got $delta_length."))
    source in (:standard, :transposed) || throw(ArgumentError(
        "source must be :standard or :transposed, got $source"))
    return source == :transposed ?
        permutedims(_delta(permutedims(x), delta_length)) :   # audioflux: along the coefficients
        _delta(x, delta_length)                               # matlab: along the frames
end

"""
    Delta(x::AbstractAudioSpectrum; delta_length=9, source=:standard) -> Delta
    Delta(x::AbstractMatrix; sr, delta_length=9, source=:standard) -> Delta

First-order temporal derivative of a feature (cepstrum, delta, spectrogram),
computed with MATLAB's `audioDelta` regression filter of odd length
`delta_length` (`DeltaWindowLength`), causal with zero initial state.
A raw matrix is taken in `frames × coeffs` orientation, like `get_data`.

- `source=:transposed` differentiates along the coefficient axis instead
  (audioflux convention).

```julia
mfcc   = Mfcc(mel; ncoeffs=13)
delta  = Delta(mfcc)
delta2 = Delta(delta)          # delta-delta
```
"""
function Delta(x::AbstractAudioSpectrum; delta_length::Int64=9, source::Symbol=:standard)
    spec = _delta_matrix(get_spec(x), delta_length, source)
    Delta{typeof(x),eltype(x)}(spec, x, DeltaSetup(get_sr(x), delta_length, source))
end

function Delta(x::AbstractMatrix{T}; sr::Int64, delta_length::Int64=9, source::Symbol=:standard) where {T<:AudioData}
    spec = _delta_matrix(permutedims(x), delta_length, source)
    Delta{Nothing,T}(spec, nothing, DeltaSetup(sr, delta_length, source))
end

"""
    DeltaDelta(x; kwargs...) -> (Delta, Delta)

Convenience returning the delta and delta-delta of `x`.
"""
function DeltaDelta(x::AbstractAudioSpectrum; kwargs...)
    d1 = Delta(x; kwargs...)
    d2 = Delta(d1; kwargs...)
    return d1, d2
end

# ---------------------------------------------------------------------------- #
#                                    methods                                   #
# ---------------------------------------------------------------------------- #
Base.eltype(::Delta{F,T}) where {F,T} = T

"""
    get_data(d::Delta) -> AbstractMatrix

Deltas transposed to `frames × coeffs` (MATLAB orientation).
"""
@inline get_data(d::Delta)   = d.spec'
@inline get_spec(d::Delta)   = d.spec
@inline get_setup(d::Delta)  = d.info
@inline get_sr(d::Delta)     = d.info.sr
@inline get_parent(d::Delta) = d.parent

function Base.show(io::IO, d::Delta{F,T}) where {F,T}
    nc, nf = size(d.spec)
    print(io, "Delta{$(nameof(F)),$T}($nf frames × $nc coeffs, length=$(d.info.delta_length))")
end
