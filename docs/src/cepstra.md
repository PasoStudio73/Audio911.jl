```@meta
CurrentModule = Audio911
```

# [Cepstra](@id cepstra)

[`Mfcc`](@ref) computes cepstral coefficients of any band spectrogram
(MATLAB's `mfcc` and `cepstralCoefficients`); [`Gtcc`](@ref) is the same
computation restricted to an [`ErbSpec`](@ref) (MATLAB's `gtcc`).
[`Delta`](@ref) is the regression-filter derivative (`audioDelta`); apply
it twice for delta-delta.

```julia
mfcc = Mfcc(mel; ncoeffs=13, rect=mlog)                # MATLAB
mfcc = Mfcc(mel; ncoeffs=13, rect=nlog, lifter=22, energy=raw_energy)   # Kaldi-like
delta, delta2 = Delta(mfcc), Delta(Delta(mfcc))
get_data(mfcc)       # frames × ncoeffs
get_ncoeffs(mfcc)
```

Keywords cover every axis of the published variants: rectification
(`mlog`, `nlog`, `cubic_root`, `db` with `top_db`), band `floor`
(`dither`), DCT scaling (`dct_ortho`, `dct_htk`, `dct_plain`), `lifter`
and `lifter_offset`, `first` coefficient kept, and a log-energy term
(`energy=raw_energy` or `spectrum_energy`, `energy_mode=:replace` or
`:append`, `energy_floor`). The presets on the [MFCC variants](@ref mfcc_variants)
page wire them for MATLAB, HTK, Kaldi, librosa, ETSI and
python_speech_features.

The input spectrogram is never modified (the floor is applied to a copy).

## xxcc

audioFlux's `xxcc` is the cepstrum of any spectrogram with `log10`
rectification, a `1e-8` floor and the orthonormal DCT:
`Mfcc(spec; rect=mlog, floor=1e-8)` computes it on an STFT, a CQT
(audioFlux's `cqcc`), a mel, bark or ERB spectrogram (`mfcc`, `bfcc`,
`gtcc`). [`xxcc_standard`](@ref) adds the log energy (replacing C0, or in
front with `energy_mode=:prepend`) and the delta and delta-delta along the
coefficient axis, audioFlux's `xxcc_standard` / `mfcc_standard`:

```julia
c, d1, d2 = xxcc_standard(Stft(audio; spectrum=magnitude); ncoeffs=13)
```
