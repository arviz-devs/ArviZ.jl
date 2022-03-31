using Documenter, Downloads, ArviZ
using MCMCChains: MCMCChains # make `from_mcmcchains` available for API docs
using SampleChains: SampleChains # make `from_samplechains` available for API docs

const ASSETS_DIR = joinpath(@__DIR__, "src", "assets")
const ARVIZ_ASSETS_URL = "https://raw.githubusercontent.com/arviz-devs/arviz_governance/main/arviz_logos"

function download_asset(remote_fn, fn=remote_fn)
    mkpath(ASSETS_DIR)
    return Downloads.download(
        joinpath(ARVIZ_ASSETS_URL, remote_fn), joinpath(ASSETS_DIR, fn); verbose=true
    )
end

# download arviz-devs org logo assets
download_asset("ArviZ.png", "logo.png")
download_asset("ArviZ_white.png", "logo-dark.png")
download_asset("ArviZ_fav.png", "favicon.png")

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
        assets=[asset("assets/favicon.png", class=:ico, islocal = true)],
        sidebar_sitename=false,
        canonical="stable",
    ),
    linkcheck=true,
    analytics="G-W1G68W77YV",
)

deploydocs(; repo="github.com/arviz-devs/ArviZ.jl.git", devbranch="main", push_preview=true)
