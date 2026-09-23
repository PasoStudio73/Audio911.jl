# ---------------------------------------------------------------------------- #
#                                 inverse STFT                                 #
# ---------------------------------------------------------------------------- #
# Weighted overlap-add (Griffin & Lim 1984), the default of audioFlux
# (stftObj_istft), librosa and MATLAB, and plain overlap-add.

"""
    istft(C, winsize, winstep; window=hanning, periodic=true, nfft=2(size(C,1)-1),
          method=:wola, offset=0, length=nothing) -> Vector
    istft(s::Stft; method=:wola, length=nothing) -> Vector

Inverse short-time Fourier transform of a one-sided complex STFT `C`
(`bins × frames`). Every column is inverse transformed (`irfft` of length
`nfft`), its first `winsize` samples are overlap-added with hop `winstep`
and the sum is normalised:

- `method=:wola` (weighted overlap-add): every frame is multiplied by the
  window and the sum is divided by `Σ w²` (Griffin–Lim; audioFlux method 0,
  librosa, MATLAB `istft`);
- `method=:ola` (overlap-add): frames are added as they are and the sum is
  divided by `Σ w` (audioFlux method 1).

Samples where the normalisation is below `1e-6` are left unnormalised.
`offset` is the index (relative to the original signal) of the first sample
of frame 1, negative for centred frames: that many samples are dropped from
the front. `length` truncates or zero-pads the result.

The second form takes the window, hop, FFT size and centring from the
`Stft` (and its `Frames`) and reconstructs the signal the frames were cut
from, after DC removal and pre-emphasis if the frames applied them. With a
window satisfying the overlap condition the reconstruction is exact.

```julia
stft = Stft(audio; winsize=1024, winstep=256, center=true, keep_complex=true)
y = istft(stft)                       # ≈ the audio samples
```
"""
function istft(C::AbstractMatrix{Complex{T}}, winsize::Int, winstep::Int;
               window::Union{Base.Callable,AbstractVector}=hanning, periodic::Bool=true,
               nfft::Int=2 * (size(C, 1) - 1), method::Symbol=:wola, offset::Int=0,
               length::Maybe{Int}=nothing) where {T<:AudioData}
    nb, nf = size(C)
    nb == nfft ÷ 2 + 1 || throw(DimensionMismatch("C has $nb bins, a one-sided STFT of nfft=$nfft has $(nfft ÷ 2 + 1)"))
    winsize ≤ nfft || throw(ArgumentError("winsize ($winsize) must be ≤ nfft ($nfft)"))
    0 < winstep || throw(ArgumentError("winstep must be positive"))
    method in (:wola, :ola) || throw(ArgumentError("method must be :wola or :ola, got :$method"))
    w = window isa AbstractVector ? Vector{T}(window) : _make_window(T, window, winsize, periodic)
    Base.length(w) == winsize || throw(DimensionMismatch("the window has $(Base.length(w)) samples, winsize is $winsize"))
    L = nf == 0 ? 0 : (nf - 1) * winstep + winsize
    y = zeros(T, L)
    norm = zeros(T, L)
    plan = plan_irfft(zeros(Complex{T}, nb), nfft)
    buf = Vector{T}(undef, nfft)
    col = Vector{Complex{T}}(undef, nb)
    @inbounds for j in 1:nf
        copyto!(col, view(C, :, j))
        mul!(buf, plan, col)
        s = (j - 1) * winstep
        if method === :wola
            for i in 1:winsize
                y[s + i] += buf[i] * w[i]
                norm[s + i] += w[i]^2
            end
        else
            for i in 1:winsize
                y[s + i] += buf[i]
                norm[s + i] += w[i]
            end
        end
    end
    @inbounds for i in eachindex(y)
        norm[i] ≥ T(1e-6) && (y[i] /= norm[i])
    end
    drop = max(-offset, 0)
    y = drop > 0 ? y[min(drop + 1, L + 1):end] : y
    if !isnothing(length)
        y = Base.length(y) ≥ length ? y[1:length] : vcat(y, zeros(T, length - Base.length(y)))
    end
    return y
end

istft(s::Stft; method::Symbol=:wola, length::Maybe{Int}=nothing) = _istft(s, get_complex(s); method, length)

# invert a complex matrix `C` on the grid of `s` (a masked or modified STFT)
function _istft(s::Stft, C::AbstractMatrix; method::Symbol, length)
    fr = get_frames(s)
    info = get_setup(fr)
    n = Base.length(get_signal(fr))
    # centred frames padded the signal by winsize ÷ 2 on both sides
    orig = info.center ? n - 2 * (info.winsize ÷ 2) : n
    len = something(length, orig)
    return istft(C, get_winsize(s), get_step(s); window=get_window(s), nfft=get_nfft(s),
                 method, offset=get_offset(s), length=len)
end
