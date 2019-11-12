using Documenter, ArviZ

makedocs(
    modules = [ArviZ],
    format = Documenter.HTML(sidebar_sitename = false),
    checkdocs = :exports,
    sitename = "ArviZ.jl",
    pages = Any["index.md"],
)

deploydocs(repo = "github.com/sethaxen/ArviZ.jl.git",)
