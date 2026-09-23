using Documenter
using Audio911

DocMeta.setdocmeta!(Audio911, :DocTestSetup, :(using Audio911); recursive = true)

makedocs(;
    modules=[Audio911],
    authors="Riccardo Pasini",
    repo=Documenter.Remotes.GitHub("aclai-lab", "Audio911.jl"),
    sitename="Audio911.jl",
    format=Documenter.HTML(;
        size_threshold=4000000,
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://aclai-lab.github.io/Audio911.jl",
        assets=String[],
    ),
    pages=[
        "Home"            => "index.md",
        "Tutorial"        => "tutorial.md",
        "Pipeline design" => "design.md",
        "Stages" => [
            "Loading audio" => "loading.md",
            "Frames"        => "frames.md",
            "STFT"          => "stft.md",
            "Wavelets"      => "cwt.md",
            "Constant-Q and other transforms" => "transforms.md",
            "Filterbanks"   => "filterbanks.md",
            "Spectrograms"  => "spectrograms.md",
            "Cepstra"       => "cepstra.md",
            "Descriptors"   => "descriptors.md",
            "Features"      => "features.md",
            "Signal utilities" => "signal.md",
            "Plotting"      => "plotting.md",
        ],
        "MFCC variants"   => "mfcc_variants.md",
        "Feature coverage" => "coverage.md",
        "audioFlux inventory" => "audioflux.md",
        "Performance"     => "performance.md",
        "API reference"   => "api.md",
    ],
    warnonly=true,
)

deploydocs(;
    repo = "github.com/aclai-lab/Audio911.jl",
    devbranch = "main",
    target = "build",
    branch = "gh-pages",
    versions = ["main" => "main", "stable" => "v^", "v#.#", "dev" => "dev"],
)
