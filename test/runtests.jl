using ArviZ
using Test

include("InferenceObjects/runtests.jl")

@testset "ArviZ" begin
    include("helpers.jl")
    include("test_rcparams.jl")
    include("test_utils.jl")
    include("test_backend.jl")
    include("test_data.jl")
    include("test_diagnostics.jl")
    include("test_stats.jl")
    include("test_plots.jl")
    include("test_samplechains.jl")
    include("test_mcmcchains.jl")
end
