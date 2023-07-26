using ArviZ, Random, Statistics, Test
using ArviZ.ArviZStats
using ArviZExampleData
using DataFrames: DataFrames
using Random

Random.seed!(97)

@testset "ArviZStats" begin
    idata = load_example_data("centered_eight")

    @testset "summarystats" begin
        rng = MersenneTwister(42)
        nchains, ndraws = 4, 10
        idata = convert_to_inference_data((
            a=randn(rng, ndraws, nchains), b=randn(rng, ndraws, nchains, 3, 4)
        ),)

        s = summarystats(idata)
        @test s isa DataFrames.DataFrame
        @test first(names(summarystats(idata))) === "variable"
        @test :variable in propertynames(summarystats(idata))
        @test "a" ∈ s.variable
        @test "b" ∉ s.variable
        @test "b[0, 0]" ∉ s.variable
        @test "b[1, 1]" ∈ s.variable
        @test s.mean != round.(s.mean; digits=1)

        s2 = summarystats(idata; digits=1)
        @test s2.mean == round.(s2.mean; digits=1)

        function median_sd(x)
            med = median(x)
            sd = sqrt(mean((x .- med) .^ 2))
            return sd
        end
        func_dict = Dict(
            "std" => x -> std(x; corrected=false),
            "median_std" => median_sd,
            "5%" => x -> quantile(x, 0.05),
            "median" => median,
            "95%" => x -> quantile(x, 0.95),
        )
        s3 = summarystats(idata; var_names=(:a,), stat_funcs=func_dict, extend=false)
        @test s3 isa DataFrames.DataFrame
        @test s3.variable == ["a"]
        @test issetequal(
            names(s3), ["variable", "std", "median_std", "5%", "median", "95%"]
        )
    end

    @testset "ArviZ.summary" begin
        rng = MersenneTwister(42)
        nchains, ndraws = 4, 10
        data = (a=randn(rng, ndraws, nchains), b=randn(rng, ndraws, nchains, 3, 4))

        @test ArviZ.summary(data) isa DataFrames.DataFrame
    end

    include("helpers.jl")
    include("utils.jl")
    include("hdi.jl")
    include("loo.jl")
    include("loo_pit.jl")
    include("waic.jl")
    include("model_weights.jl")
    include("compare.jl")
    include("r2_score.jl")
end
