# ---------------------------------------------------------------------------- #
#                          harmonic count (audioFlux)                          #
# ---------------------------------------------------------------------------- #
# Port of audioFlux's mir/harmonic_algorithm.c (MIT licence, Copyright (c)
# 2023 libAudioFlux). The peak picking and its three filters are heuristics
# with fixed constants; they are translated literally (0-based offsets kept
# in the comments' sense), including the in-place compactions that read
# already-rewritten neighbours, so the counts match audioFlux's.

# audioFlux __vcorrsort1: exchange sort of `a` over `r`, permuting b, c, d alike
function _hc_corrsort!(a, b, c, d, r::UnitRange{Int}, desc::Bool)
    @inbounds for i in r, j in i+1:last(r)
        if desc ? a[i] < a[j] : a[i] > a[j]
            a[i], a[j] = a[j], a[i]; b[i], b[j] = b[j], b[i]
            c[i], c[j] = c[j], c[i]; d[i], d[j] = d[j], d[i]
        end
    end
end

# peaks of one frame: dB-sorted (db, fre, height, idx) and their count;
# P and D are the power and dB of bins minIndex…maxIndex (1-based here)
function _hc_peaks!(db, fre, ht, ix, P::AbstractVector{T}, D::AbstractVector{T},
                    minIndex::Int, nfft::Int, sr::Int) where T
    minHeight, cutDB = T(15), T(-50)
    rLen = length(P)
    len = 0
    j = 1
    @inbounds while j < rLen - 1
        pre, cur, nex = P[j], P[j + 1], P[j + 2]
        if cur > pre && cur > nex
            idx = j + 1
            xFlag = false; eFlag = false
            f = T((j + minIndex) / nfft * sr)
            d = D[j + 1]
            pre, cur, nex = D[j], D[j + 1], D[j + 2]
            left = pre
            if j - 2 ≥ 0
                left = D[j - 1]
                if left < pre || (left > pre && left < cur && left - pre < 2 && cur > cutDB)
                    if j - 3 ≥ 0
                        pre = D[j - 2]
                        if pre < left
                            left = pre
                            if D[j - 1] > D[j] && D[j - 1] < cur && D[j - 1] - D[j] < 2
                                xFlag = true
                            end
                            if j - 4 ≥ 0 && d - left < minHeight && cur > cutDB
                                if D[j - 3] < pre
                                    left = D[j - 3]
                                    eFlag = true
                                end
                            end
                        end
                    end
                else
                    left = pre
                end
            end
            right = nex
            if j + 2 < rLen
                right = D[j + 3]
                if right < nex || (right > nex && right < cur && right - nex < 2 && cur > cutDB)
                    if j + 3 < rLen
                        nex = D[j + 4]
                        if nex < right
                            right = nex
                            idx = j + 3
                            if j + 4 < rLen && d - right < minHeight && !eFlag && cur > cutDB
                                if D[j + 5] < nex
                                    right = D[j + 5]
                                    idx = j + 4
                                end
                            end
                        else
                            idx = j + 2
                        end
                    end
                else
                    right = nex
                    idx = j + 1
                end
            end
            h1, h2 = d - left, d - right
            height = min(h1, h2)
            if height > minHeight && xFlag && h1 < h2
                # replaces the previous peak (audioFlux writes out of the
                # frame when there is none)
                if len ≥ 1
                    db[len] = d; fre[len] = f; ht[len] = height; ix[len] = j
                end
            else
                len += 1
                db[len] = d; fre[len] = f; ht[len] = height; ix[len] = j
            end
            j = idx
        end
        j += 1
    end
    _hc_corrsort!(db, fre, ht, ix, 1:len, true)
    return len
end

# the height, near and dB filters of one frame; returns the frequencies kept
function _hc_filter(db::Vector{T}, fre, ht, ix, len::Int, maxDB::T) where T
    minHeight = T(15)
    n = length(db)
    # height filter (the two loudest peaks always pass)
    db1 = zeros(T, n); fr1 = zeros(T, n); ht1 = zeros(T, n); ix1 = zeros(Int, n)
    start, len1 = len ≥ 2 ? (2, 2) : len ≥ 1 ? (1, 1) : (0, 0)
    firstIndex = len1 ≥ 1 ? ix[1] : 0
    secondIndex = len1 ≥ 2 ? ix[2] : 0
    @inbounds for k in 1:len1
        db1[k] = db[k]; fr1[k] = fre[k]; ht1[k] = ht[k]; ix1[k] = ix[k]
    end
    _hc_corrsort!(fre, db, ht, ix, start+1:len, false)
    @inbounds for j in start:len-1          # 0-based j, element j + 1
        ht[j + 1] > minHeight || continue
        curDb, preDb, nexDb = db[j + 1], db[j], db[j + 2]
        preH, nexH = ht[j], ht[j + 2]
        curI, preI, nexI = ix[j + 1], ix[j], ix[j + 2]
        firstIndex != 0 && firstIndex > preI && firstIndex < curI && (preH = minHeight + 1)
        secondIndex != 0 && secondIndex > preI && secondIndex < curI && (preH = minHeight + 1)
        firstIndex != 0 && firstIndex > curI && firstIndex < nexI && (nexH = minHeight + 1)
        secondIndex != 0 && secondIndex > curI && secondIndex < nexI && (nexH = minHeight + 1)
        if (curDb - preDb > 12 || preH > minHeight) && (curDb - nexDb > 12 || nexH > minHeight)
            len1 += 1
            db1[len1] = db[j + 1]; fr1[len1] = fre[j + 1]; ht1[len1] = ht[j + 1]; ix1[len1] = ix[j + 1]
        end
    end
    _hc_corrsort!(fr1, db1, ht1, ix1, 1:len1, false)
    # near filter (peaks closer than 30 Hz)
    minFre = T(30)
    db2 = zeros(T, n); fr2 = zeros(T, n)
    len2 = 0; lastFlag = true
    j = 0
    @inbounds while j < len1 - 1
        k = j
        if fr1[j + 2] - fr1[j + 1] < minFre
            j == len1 - 2 && (lastFlag = false)
            if db1[j + 1] < db1[j + 2]
                k = j + 1
                if j + 2 < len1 && fr1[j + 3] - fr1[j + 2] < minFre && db1[j + 2] > db1[j + 3]
                    j += 1
                end
            end
            j += 1
        end
        len2 += 1
        db2[len2] = db1[k + 1]; fr2[len2] = fr1[k + 1]
        j += 1
    end
    if lastFlag && len1 ≥ 1
        len2 += 1
        db2[len2] = db1[len1]; fr2[len2] = fr1[len1]
    elseif lastFlag
        len2 += 1                                # audioFlux copies slot -1 (zeros)
    end
    # dB filter
    minDB = T(15)
    db3 = zeros(T, n); fr3 = zeros(T, n)
    len3 = 0
    @inbounds for k in 1:len2
        if db2[k] > -100
            len3 += 1; db3[len3] = db2[k]; fr3[len3] = fr2[k]
        end
    end
    m = 0
    j = 0
    @inbounds while j < len3
        m += 1
        db3[m] = db3[j + 1]; fr3[m] = fr3[j + 1]
        if j + 3 < len3
            d1, d2, d3, d4 = db3[j + 1], db3[j + 2], db3[j + 3], db3[j + 4]
            if d1 - d2 > minDB && d1 - d3 > minDB && d4 - d2 > minDB && d4 - d3 > minDB
                j += 2
            end
        end
        j += 1
    end
    len2 = m
    len3 = 0
    st = 0
    top = len2 == 0 ? 0 : argmax(view(db3, 1:len2)) - 1
    @inbounds for j in 0:top
        if maxDB - db3[j + 1] < minDB || db3[j + 1] > -42
            st = j
            len3 += 1
            db3[len3] = db3[j + 1]; fr3[len3] = fr3[j + 1]
        end
    end
    @inbounds for j in st+1:len2-2
        if db3[j] - db3[j + 1] < minDB || db3[j + 2] - db3[j + 1] < minDB
            len3 += 1
            db3[len3] = db3[j + 1]; fr3[len3] = fr3[j + 1]
        end
    end
    @inbounds if len2 > 1 && st < len2 - 1
        if db3[len2 - 1] - db3[len2] < minDB || len2 == 3 || len3 == 2
            len3 += 1
            db3[len3] = db3[len2]; fr3[len3] = fr3[len2]
        end
    end
    return fr3[1:len3]
end

"""
    harmonic_count(stft::Stft; range=(27, 4000), count_range=range) -> Vector{Int}

Number of harmonic peaks in every frame of a power STFT (audioFlux
`Harmonic.harmonic_count`). The spectral peaks between the bins of
`range` are located on the dB spectrum `10 log10(|X|² / nfft²)`, merged
with shoulders of less than 2 dB and kept by three heuristic filters: a
height filter (peaks at least 15 dB above their neighbourhood, or 12 dB
above the adjacent peaks; the two loudest always pass), a proximity filter
(of two peaks closer than 30 Hz the louder one) and a level filter (peaks
within 15 dB of the loudest or above -42 dB, isolated dips removed). The
peaks strictly inside `count_range` (Hz) are counted.

The absolute levels assume the plain unscaled STFT (`Stft(frames)`);
audioFlux's default is a Hamming window of 4096 samples with a hop of 1024:
`harmonic_count(Stft(Frames(audio; winsize=4096, winstep=1024, type=hamming)))`.
"""
function harmonic_count(s::Stft; range::FreqRange=(27, 4000), count_range::FreqRange=range)
    T = eltype(s)
    sr = get_sr(s); nfft = get_nfft(s)
    lo, hi = get_low(range), get_hi(range)
    0 ≤ lo < hi || throw(ArgumentError("range must satisfy 0 ≤ low < high, got $range"))
    minIndex = floor(Int, T(lo) * nfft / sr)
    maxIndex = ceil(Int, T(hi) * nfft / sr)
    maxIndex ≥ nfft ÷ 2 && (maxIndex = nfft ÷ 2 - 1)
    minIndex < maxIndex || throw(ArgumentError("range $range holds no bins"))
    clo, chi = T(get_low(count_range)), T(get_hi(count_range))
    X = get_spec(s)
    pw = get_spectrum(s) === power
    rLen = maxIndex - minIndex + 1
    peakLength = (maxIndex - minIndex) ÷ 2 + 1
    P = Vector{T}(undef, rLen); D = similar(P)
    db = zeros(T, peakLength + 1); fre = similar(db); ht = similar(db); ix = zeros(Int, peakLength + 1)
    counts = zeros(Int, get_nframes(s))
    for t in eachindex(counts)
        @inbounds for k in 1:rLen
            P[k] = pw ? X[minIndex + k, t] : X[minIndex + k, t]^2
            D[k] = 10 * log10(P[k] / nfft / nfft)
        end
        fill!(db, zero(T)); fill!(fre, zero(T)); fill!(ht, zero(T)); fill!(ix, 0)
        len = _hc_peaks!(db, fre, ht, ix, P, D, minIndex, nfft, sr)
        maxDB = db[1]
        for f in _hc_filter(db, fre, ht, ix, len, maxDB)
            f ≥ chi && break
            f > clo && (counts[t] += 1)
        end
    end
    return counts
end
