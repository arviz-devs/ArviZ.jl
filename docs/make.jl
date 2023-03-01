using Documenter, Downloads, ArviZ
using MCMCChains: MCMCChains # make `from_mcmcchains` available for API docs
using SampleChains: SampleChains # make `from_samplechains` available for API docs
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

# download arviz-devs org logo assets
download_asset("ArviZ.png", "logo.png")
download_asset("ArviZ_white.png", "logo-dark.png")
download_asset("favicon.ico")

DocMeta.setdocmeta!(
    ArviZExampleData,
    :DocTestSetup,
    :(using ArviZ.ArviZExampleData, ArviZ.InferenceObjects);
    recursive=true,
)

makedocs(;
    modules=[
        ArviZ,
        ArviZExampleData,
        InferenceObjects,
        InferenceObjectsNetCDF,
        MCMCDiagnosticTools,
        PSIS,
    ],
    sitename="ArviZ.jl",
    pages=[
        "Home" => "index.md",
        "Getting Started" => [
            "Quickstart" => "quickstart.md",
            "Working with `InferenceData`" => "working_with_inference_data.md",
            "Creating custom plots" => "creating_custom_plots.md",
        ],
        "Example Gallery" => ["Matplotlib" => "mpl_examples.md"],
        "API" => [
            hide("api/index.md"),
            "Plots" => "api/plots.md",
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
    linkcheck=true,
    analytics="G-W1G68W77YV",
)

deploydocs(; repo="github.com/arviz-devs/ArviZ.jl.git", devbranch="main", push_preview=true)
