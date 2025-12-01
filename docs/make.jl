using Documenter, DocumenterInterLinks, Downloads, ArviZ
using PlutoStaticHTML: PlutoStaticHTML

const DOCS_SRC_PATH = joinpath(@__DIR__, "src")

# generate markdown from Pluto notebooks
output_format = PlutoStaticHTML.documenter_output
build_opts = PlutoStaticHTML.BuildOptions(
    DOCS_SRC_PATH;
    previous_dir=DOCS_SRC_PATH,
    output_format=output_format,
    add_documenter_css=false,
)
PlutoStaticHTML.build_notebooks(build_opts)

const ASSETS_DIR = joinpath(@__DIR__, "src", "assets")
const ARVIZ_ASSETS_URL = "https://raw.githubusercontent.com/arviz-devs/arviz-project/main/arviz_logos"

function download_asset(remote_fn, fn=remote_fn)
    mkpath(ASSETS_DIR)
    return Downloads.download(
        joinpath(ARVIZ_ASSETS_URL, remote_fn), joinpath(ASSETS_DIR, fn); verbose=true
    )
end

function get_extension(mod::Module, submodule::Symbol)
    if isdefined(Base, :get_extension)
        return Base.get_extension(mod, submodule)
    else
        return getproperty(mod, submodule)
    end
end

# download arviz-devs org logo assets
download_asset("ArviZ.png", "logo.png")
download_asset("ArviZ_white.png", "logo-dark.png")
download_asset("favicon.ico")

InferenceObjectsMCMCDiagnosticToolsExt = get_extension(
    InferenceObjects, :InferenceObjectsMCMCDiagnosticToolsExt
)
InferenceObjectsPosteriorStatsExt = get_extension(
    InferenceObjects, :InferenceObjectsPosteriorStatsExt
)

for subpkg in (InferenceObjects, MCMCDiagnosticTools, PosteriorStats, PSIS)
    DocMeta.setdocmeta!(subpkg, :DocTestSetup, :(using $(Symbol(subpkg))))
end
DocMeta.setdocmeta!(
    InferenceObjectsMCMCDiagnosticToolsExt, :DocTestSetup, :(using MCMCDiagnosticTools)
)
DocMeta.setdocmeta!(
    InferenceObjectsPosteriorStatsExt, :DocTestSetup, :(using PosteriorStats)
)

links = InterLinks(
    "arviz" => "https://python.arviz.org/en/stable/",
    "DimensionalData" => "https://rafaqz.github.io/DimensionalData.jl/stable/",
    "IntervalSets" => "https://juliamath.github.io/IntervalSets.jl/stable/",
    "MCMCDiagnosticTools" => "https://julia.arviz.org/MCMCDiagnosticTools/stable/",
    "MLJ" => "https://juliaai.github.io/MLJ.jl/stable/",
    "NCDatasets" => "https://juliageo.org/NCDatasets.jl/stable/",
    "PosteriorStats" => "https://julia.arviz.org/PosteriorStats/stable/",
    "PSIS" => "https://julia.arviz.org/PSIS/stable/",
    "Statistics" => "https://docs.julialang.org/en/v1/",
    "StatsBase" => "https://juliastats.org/StatsBase.jl/stable/",
)

modules = [
    ArviZ,
    InferenceObjects,
    InferenceObjectsMCMCDiagnosticToolsExt,
    InferenceObjectsPosteriorStatsExt,
    MCMCDiagnosticTools,
    PosteriorStats,
    PSIS,
]

doctestfilters = [
    r"\s+\"created_at\" => .*",  # ignore timestamps in doctests,
    r"(└.*packages/[A-Za-z]*/).*" => s"\1",  # ignore package paths in warning messages
]

prettyurls = haskey(ENV, "CI")

warnonly = [
    :missing_docs,  # we don't reexport or document all of MCMCDiagnosticTools
    :cross_references,  # we reference the pages built by PlutoStaticHTML with pretty relative URLs
    :linkcheck,  # avoid checking links to pages built with PlutoStaticHTML
    :external_cross_references,  # rely on our component packages to get external cross-references right
    :doctest,  # rely on our component packages to get doctests right
]

withenv("COLUMNS" => 100) do
    makedocs(;
        modules,
        sitename="ArviZ.jl",
        pages=[
            "Home" => "index.md",
            "Getting Started" => [
                "Quickstart" => "quickstart.md",
                "Working with `InferenceData`" => "working_with_inference_data.md",
                "Creating custom plots" => "creating_custom_plots.md",
            ],
            "API" => [
                hide("api/index.md"),
                "Stats" => "api/stats.md",
                "Diagnostics" => "api/diagnostics.md",
                "Data" => "api/data.md",
                "InferenceObjects" => [
                    "InferenceData" => "api/inference_data.md",
                    "Dataset" => "api/dataset.md",
                ],
            ],
        ],
        checkdocs=:exports,
        format=Documenter.HTML(;
            prettyurls,
            size_threshold=2^21,  # 2Mb, needed since PlutoStaticHTML embeds images in markdown
            assets=["assets/favicon.ico", "assets/custom.css"],
            sidebar_sitename=false,
            canonical="stable",
            analytics="G-W1G68W77YV",
        ),
        doctestfilters,
        linkcheck=true,
        warnonly,
        plugins=[links],
    )
end

deploydocs(; repo="github.com/arviz-devs/ArviZ.jl.git", devbranch="main", push_preview=true)
