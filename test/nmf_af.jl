using Test
using Audio911
using MAT

af_files_dir()    = joinpath(dirname(@__FILE__), "audioflux_files", "nmf")
af_file(filename) = joinpath(af_files_dir(), filename)
rel_l2(a, b) = sqrt(sum(abs2, a .- b)) / sqrt(sum(abs2, b))

@testset "nmf" begin
    # an exactly factorisable matrix with three components
    Wt = [exp(-((i - c) / 6)^2) for i in 1:60, c in (12, 30, 48)]
    Ht = [max(0, sin(0.2j + 2c)) + 0.05 for c in 1:3, j in 1:80]
    V = Wt * Ht
    for div in (:kl, :is, :euclidean)
        W, H = nmf(V, 3; divergence=div, max_iter=2000, thresh=1e-9)
        @test size(W) == (60, 3) && size(H) == (3, 80)
        @test all(≥(0), W) && all(≥(0), H)
        @test rel_l2(W * H, V) < 0.02
        @test maximum(W; dims=1) ≈ ones(1, 3)          # norm=:max
        # every true template is found by one component
        C = [sum(Wt[:, a] .* W[:, b]) / sqrt(sum(abs2, Wt[:, a]) * sum(abs2, W[:, b])) for a in 1:3, b in 1:3]
        @test all(maximum(C; dims=2) .> 0.98)
    end
    W, H = nmf(V, 3; norm=:l1)
    @test vec(sum(W; dims=1)) ≈ ones(3)
    W, H = nmf(V, 3; norm=:l2)
    @test vec(sqrt.(sum(abs2, W; dims=1))) ≈ ones(3)
    # a user initialisation, Float32 and the spectrogram method
    W0, H0 = fill(0.5, 60, 3), fill(0.5, 3, 80)
    @test size(nmf(V, 3; init=(W0, H0))[1]) == (60, 3)
    W32, H32 = nmf(Float32.(V), 3)
    @test eltype(W32) == Float32 && eltype(H32) == Float32
    s = Stft(Frames(sin.(2π * 440 .* (0:7999) ./ 8000), 8000; winsize=256, winstep=128); spectrum=magnitude)
    Ws, Hs = nmf(s, 2)
    @test size(Ws) == (129, 2) && size(Hs) == (2, get_nframes(s))
    @test_throws ArgumentError nmf(V, 0)
    @test_throws ArgumentError nmf(V, 3; divergence=:foo)
    @test_throws ArgumentError nmf(V, 3; norm=:foo)
    @test_throws ArgumentError nmf(-V, 3)
    @test_throws ArgumentError nmf(V, 3; init=:random)
    @test_throws DimensionMismatch nmf(V, 3; init=(W0, fill(0.5, 2, 80)))
end

@testset "nmf against audioFlux" begin
    mat = MAT.matread(af_file("nmf.mat"))
    X = Float64.(mat["X"])
    # h, w = audioflux.classic.nmf.nmf(X, 4, max_iter=<iters>, tp=<tp>, thresh=1e-3, norm=<norm>)
    for (tp, div) in ((0, :kl), (1, :is), (2, :euclidean)), (nm, norm) in ((0, :max), (1, :l1), (2, :l2)),
        iters in (20, 300)
        W, H = nmf(X, 4; divergence=div, norm, max_iter=iters, thresh=1e-3, init=:audioflux)
        key = "$(tp)_$(nm)_$(iters)"
        rw, rh = rel_l2(W, mat["w_" * key]), rel_l2(H, mat["h_" * key])
        tol = 1e-5                          # float32 agreement
        @test rw < tol
        @test rh < tol
        rw < tol && rh < tol || @info "nmf $key" rw rh
    end
end
