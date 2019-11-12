using Documenter, ArviZ

makedocs(
    modules = [ArviZ],
    sitename = "ArviZ.jl",
    pages = [
        "Home" => "index.md",
        "Quickstart" => "quickstart.md",
    ],
    checkdocs = :exports,
    format = Documenter.HTML(
        prettyurls = haskey(ENV, "CI"),
        sidebar_sitename = false,
    ),
)

deploydocs(repo = "github.com/sethaxen/ArviZ.jl.git",)
