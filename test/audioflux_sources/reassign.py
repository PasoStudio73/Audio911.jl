"""audioFlux STFT, ISTFT and reassignment reference outputs (test/audioflux_files/reassign/).

Julia counterpart: test/reassign.jl. Frames of 512 samples (radix2_exp=9),
hop 128, periodic Hann window, no padding.
"""
import numpy as np
import audioflux as af
from audioflux.type import WindowType, ReassignType
from _common import load, save

x, sr = load()

# stft: obj = af.STFT(radix2_exp=9, window_type=WindowType.HANN, slide_length=128); S = obj.stft(x)
obj = af.STFT(radix2_exp=9, window_type=WindowType.HANN, slide_length=128)
S = obj.stft(x)
# istft of the full two-sided spectrum (the wrapper expects fft_length//2+1 rows)
y0 = obj.istft(S, method_type=0)
y1 = obj.istft(S, method_type=1)
save("reassign", "stft", stft=S, istft_wola=y0, istft_ola=y1)

cases = {
    # name: (re_type, order, result_type)
    "all":        (ReassignType.ALL, 1, 0),
    "fre":        (ReassignType.FRE, 1, 0),
    "time":       (ReassignType.TIME, 1, 0),
    "all_order2": (ReassignType.ALL, 2, 0),
    "all_mag":    (ReassignType.ALL, 1, 1),
}
for name, (re_type, order, result_type) in cases.items():
    # obj = af.Reassign(radix2_exp=9, samplate=sr, window_type=WindowType.HANN, slide_length=128,
    #                   re_type=<re_type>, thresh=0.001, is_padding=False)
    # obj.set_order(order); re, S = obj.reassign(x, result_type=result_type)
    obj = af.Reassign(radix2_exp=9, samplate=sr, window_type=WindowType.HANN, slide_length=128,
                      re_type=re_type, thresh=0.001, is_padding=False)
    if order > 1:
        obj.set_order(order)
    re, S = obj.reassign(x, result_type=result_type)
    save("reassign", name, re=re, stft=S)
