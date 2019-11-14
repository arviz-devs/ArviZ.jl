using Pkg;
Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()
Pkg.activate(); Pkg.instantiate()

pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

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
        analytics = "UA-152384573-1",
    ),
)

deploydocs(repo = "github.com/arviz-devs/ArviZ.jl.git",)
