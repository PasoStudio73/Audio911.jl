# Constant-Q chromagram (audioFlux src/filterbank/chroma_filterBank.c,
# chroma_cqtFilterBank; MIT licence, Copyright (c) 2023 libAudioFlux).
# ---------------------------------------------------------------------------- #
#                                  cqt chroma                                  #
# ---------------------------------------------------------------------------- #
"""
    cqt_chroma_fbank(c::Cqt; nchroma=12) -> ChromaFBank
    cqt_chroma_fbank(freq, bins_per_octave; nchroma=12, sr) -> ChromaFBank

Folding matrix of a constant-Q grid onto `nchroma` pitch classes (audioFlux
`chroma_cqtFilterBank`): every bin belongs to exactly one class, the
`bins_per_octave ÷ nchroma` bins nearest to each class centre, and row 1 is
the pitch class C whatever `fmin` is. `bins_per_octave` must be a multiple
of `nchroma`.
"""
function cqt_chroma_fbank(freq::AbstractVector{T}, bins_per_octave::Int; nchroma::Int=12, sr::Int=0) where {T<:AudioData}
    nchroma ≥ 1 || throw(ArgumentError("nchroma must be ≥ 1"))
    bins_per_octave % nchroma == 0 || throw(ArgumentError(
        "bins_per_octave ($bins_per_octave) must be a multiple of nchroma ($nchroma)"))
    n  = bins_per_octave ÷ nchroma
    # position of the first bin on the grid of `bins_per_octave` steps from C0
    p0 = round(Int, bins_per_octave * log2(Float64(freq[1]) / midi_to_hz(0)))
    W  = zeros(T, nchroma, length(freq))
    for k in eachindex(freq)
        kabs = p0 + k - 1
        cls  = mod(fld(2kabs + n, 2n), nchroma)          # nearest class centre
        W[cls + 1, k] = one(T)
    end
    return ChromaFBank{T}(W, Vector{T}(freq), ChromaFBankSetup(sr, nchroma, 0.0, 0.0, nothing, true))
end
cqt_chroma_fbank(c::Cqt; nchroma::Int=12) =
    cqt_chroma_fbank(get_freq(c), c.info.bins_per_octave; nchroma, sr=get_sr(c))

"""
    Chroma(c::Cqt; nchroma=12, norm=Inf) -> Chroma

Constant-Q chromagram (audioFlux `CQT.chroma`, librosa `chroma_cqt` without
its smoothing): the bins of every octave are summed into `nchroma` pitch
classes with [`cqt_chroma_fbank`](@ref) and every frame is normalised by
its `norm`-norm (`Inf` = maximum, `1`, `2`, or `nothing`).
"""
Chroma(c::Cqt; nchroma::Int=12, norm::Maybe{Real}=Inf) = Chroma(c, cqt_chroma_fbank(c; nchroma); norm)
