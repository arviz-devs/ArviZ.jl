using Documenter, ArviZ

makedocs(
    modules = [ArviZ],
    sitename = "ArviZ.jl",
    pages = [
        "Home" => "index.md",
        "Quickstart" => "quickstart.md",
        "API" => "api.md",
        "Example Gallery" => [
            "Matplotlib" => "mpl_examples.md",
            "Bokeh" => "bokeh_examples.md",
        ],
        "Reference" => "reference.md",
    ],
    checkdocs = :exports,
    format = Documenter.HTML(
        prettyurls = haskey(ENV, "CI"),
        assets = ["assets/favicon.ico"],
        sidebar_sitename = false,
        canonical = "stable",
    ),
    linkcheck = true,
)

deploydocs(repo = "github.com/arviz-devs/ArviZ.jl.git", push_preview = true)
