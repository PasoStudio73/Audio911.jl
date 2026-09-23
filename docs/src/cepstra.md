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
