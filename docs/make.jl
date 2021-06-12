using Documenter, ArviZ
using MCMCChains: MCMCChains # make `from_mcmcchains` available for API docs
using Pluto

function build_notebook(
    nbpath, htmlpath="$(first(splitext(htmlpath))).html"; project=DOCS_PATH
)
    @info "Building notebook at $nbpath"
    s = Pluto.ServerSession()
    nb = Pluto.SessionActions.open(s, nbpath; run_async=false)
    write(htmlpath, Pluto.generate_html(nb))
    return htmlpath
end

const DOCS_PATH = @__DIR__
const SRC_PATH = joinpath(DOCS_PATH, "src")
const NB_PATH = joinpath(DOCS_PATH, "notebooks")

# build Pluto notebooks
notebooks = [
    joinpath(NB_PATH, "quickstart.jl") => joinpath(SRC_PATH, "quickstart_notebook.html")
]
map(notebooks) do (nbpath, htmlpath)
    build_notebook(nbpath, htmlpath)
end

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

# cleanup built Pluto notebooks
map(notebooks) do (_, htmlpath)
    rm(htmlpath; force=true)
end

deploydocs(; repo="github.com/arviz-devs/ArviZ.jl.git", devbranch="main", push_preview=true)
