"""Regenerate src/wavelet/dwt_coefs.jl from an audioFlux source checkout.

Usage: AUDIOFLUX_SRC=/path/to/audioFlux python test/audioflux_sources/_gen_dwt_coefs.py
(not run by `test/run.sh oracle`: the leading underscore excludes it).
"""
import re, collections
import os, sys
AF = os.path.join(os.environ.get("AUDIOFLUX_SRC", "audioFlux-master"), "src/filterbank/coef/")
arrays = {}
for fn in ("__coef_d.h", "__coef_r.h"):
    txt = open(AF + fn).read()
    for m in re.finditer(r'float __([a-z0-9]+)_(lo|hi)([DR])\[(\d+)\]\s*=\s*\{([^}]*)\}', txt):
        name, band, kind, n, body = m.groups()
        vals = [v.strip() for v in body.split(',') if v.strip()]
        assert len(vals) == int(n), (name, band, kind, n, len(vals))
        arrays[(name, band + kind)] = vals
names = []
for (name, key) in arrays:
    if name not in names:
        names.append(name)
def pretty(name):
    m = re.fullmatch(r'bior(\d)(\d)', name)
    return f"bior{m.group(1)}.{m.group(2)}" if m else name
out = ["# ---------------------------------------------------------------------------- #",
       "#                     discrete wavelet filter coefficients                     #",
       "# ---------------------------------------------------------------------------- #",
       "# Decomposition (loD, hiD) and reconstruction (loR, hiR) filters of the",
       "# discrete wavelets, transcribed from audioFlux",
       "# (src/filterbank/coef/__coef_d.h and __coef_r.h, MIT licence,",
       "# Copyright (c) 2023 libAudioFlux) with their six significant decimals.",
       "# Generated; do not edit by hand.",
       "",
       "const _DWT_FILTERS = Dict{String,NTuple{4,Vector{Float64}}}("]
for name in names:
    f = [arrays[(name, k)] for k in ("loD", "hiD", "loR", "hiR")]
    out.append(f'    "{pretty(name)}" => (')
    for vals in f:
        out.append("        [" + ", ".join(vals) + "],")
    out.append("    ),")
out.append(")")
out.append("")
out.append("const DISCRETE_WAVELETS = (" + ", ".join(f'"{pretty(n)}"' for n in names) + ",)")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "wavelet", "dwt_coefs.jl"), "w").write("\n".join(out) + "\n")
print(len(names), "wavelets:", ", ".join(pretty(n) for n in names))
