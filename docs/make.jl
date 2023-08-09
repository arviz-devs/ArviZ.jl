using Documenter, Downloads, ArviZ
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
    r"\s+\"created_at\" => .*",  # ignore timestamps in doctests
]

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
        prettyurls=haskey(ENV, "CI"),
        assets=["assets/favicon.ico", "assets/custom.css"],
        sidebar_sitename=false,
        canonical="stable",
    ),
    doctestfilters,
    linkcheck=true,
    analytics="G-W1G68W77YV",
    # allow linkcheck to fail so we can use pretty links to PlutoStaticHTML notebooks
    strict=Documenter.except(:footnote, :linkcheck, :missing_docs),
)

deploydocs(; repo="github.com/arviz-devs/ArviZ.jl.git", devbranch="main", push_preview=true)
