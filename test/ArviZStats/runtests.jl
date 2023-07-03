using ArviZ, Random, Statistics, Test
using ArviZ.ArviZStats
using DataFrames: DataFrames

@testset "ArviZStats" begin
    idata = load_example_data("centered_eight")

    @testset "compare" begin
        idata2 = load_example_data("non_centered_eight")
        model_dict = Dict("a" => idata, "b" => idata2)
        loo_dict = Dict("a" => loo(idata), "b" => loo(idata2))
        df = compare(model_dict)
        @test df isa DataFrames.DataFrame
        df2 = compare(loo_dict)
        @test df2 isa DataFrames.DataFrame
        @test_broken df == df2
    end

    @testset "hdi" begin
        rng = Random.MersenneTwister(42)
        x = randn(rng, 100)
        @test hdi(x) == ArviZ.arviz.hdi(x)
    end

    @testset "r2_score" begin
        rng = Random.MersenneTwister(42)
        ytrue = randn(rng, 100)
        ypred = randn(rng, 100)
        df = r2_score(ytrue, ypred)
        @test df isa DataFrames.DataFrame
        @test all(df == ArviZ.todataframes(ArviZ.arviz.r2_score(ytrue, ypred)))
    end

    @testset "loo" begin
        df = loo(idata)
        @test df isa ArviZStats.PSISLOOResult
    end

    @testset "waic" begin
        df = waic(idata)
        @test df isa ArviZStats.WAICResult
    end

    @testset "loo_pit" begin
        ret = loo_pit(idata; y="obs")
        @test ret ≈ ArviZ.arviz.loo_pit(idata; y="obs") rtol = 0.02
    end

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
    include("loo.jl")
    include("waic.jl")
end
