using ArviZ
using Test

@testset "ArviZ" begin
    include("helpers.jl")
    include("test_utils.jl")
    include("ArviZStats/runtests.jl")
    include("test_samplechains.jl")
    include("test_mcmcchains.jl")
end
