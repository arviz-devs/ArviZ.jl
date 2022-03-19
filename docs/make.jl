using Documenter, ArviZ
using MCMCChains: MCMCChains # make `from_mcmcchains` available for API docs
using SampleChains: SampleChains # make `from_samplechains` available for API docs
using PlutoStaticHTML: PlutoStaticHTML

const DOCS_SRC_PATH = joinpath(@__DIR__, "src")

# generate markdown from Pluto notebooks
output_format = PlutoStaticHTML.documenter_output
build_opts = PlutoStaticHTML.BuildOptions(
    DOCS_SRC_PATH; previous_dir=DOCS_SRC_PATH, output_format=output_format
)
PlutoStaticHTML.build_notebooks(build_opts)

makedocs(;
    modules=[ArviZ],
    sitename="ArviZ.jl",
    pages=[
        "Home" => "index.md",
        "Quickstart" => "quickstart.md",
        "API" => "api.md",
        "Example Gallery" =>
            ["Matplotlib" => "mpl_examples.md", "Bokeh" => "bokeh_examples.md"],
        "Reference" => "reference.md",
    ],
    checkdocs=:exports,
    format=Documenter.HTML(;
        prettyurls=haskey(ENV, "CI"),
        assets=["assets/favicon.ico"],
        sidebar_sitename=false,
        canonical="stable",
    ),
    linkcheck=true,
    analytics="G-W1G68W77YV",
)

deploydocs(; repo="github.com/arviz-devs/ArviZ.jl.git", devbranch="main", push_preview=true)
