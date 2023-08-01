using ArviZ, Random, Statistics, Test
using ArviZ.ArviZStats
using ArviZExampleData
using Random

Random.seed!(97)

@testset "ArviZStats" begin
    include("helpers.jl")
    include("utils.jl")
    include("hdi.jl")
    include("loo.jl")
    include("loo_pit.jl")
    include("waic.jl")
    include("model_weights.jl")
    include("compare.jl")
    include("r2_score.jl")
    include("summarystats.jl")
end
