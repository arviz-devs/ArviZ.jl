using DataFrames: DataFrames

@testset "stats" begin
    idata = load_arviz_data("centered_eight")

    @testset "compare" begin
        idata2 = load_arviz_data("non_centered_eight")
        df = compare(Dict("a" => idata, "b" => idata2))
        @test df isa DataFrames.DataFrame
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
        df = loo(idata; pointwise=false)
        @test df isa DataFrames.DataFrame
        @test all(df == ArviZ.todataframes(ArviZ.arviz.loo(idata; pointwise=false)))
    end

    @testset "psislw" begin
        @testset for sz in ((1000,), (10, 1000))
            logw = randn(sz)
            logw_smoothed, k = psislw(copy(logw), 0.9)

            # check against PSIS.jl
            result = psis(copy(logw), 0.9)
            @test exp.(logw_smoothed) ≈ result.weights
            @test k ≈ result.pareto_shape
        end
    end

    @testset "waic" begin
        df = waic(idata; pointwise=false)
        @test df isa DataFrames.DataFrame
        @test all(df == ArviZ.todataframes(ArviZ.arviz.waic(idata; pointwise=false)))
    end

    @testset "loo_pit" begin
        ret = loo_pit(idata; y="obs")
        @test ret == ArviZ.arviz.loo_pit(idata; y="obs")
    end

    @testset "summarystats" begin
        rng = MersenneTwister(42)
        nchains, ndraws = 4, 10
        idata = convert_to_inference_data(
            Dict(
                "a" => randn(rng, nchains, ndraws), "b" => randn(rng, nchains, ndraws, 3, 4)
            ),
        )

        s = summarystats(idata)
        @test s isa DataFrames.DataFrame
        @test string(first(names(summarystats(idata)))) == "variable"
        @test string(first(names(summarystats(idata; fmt="wide")))) == "variable"
        @test :variable in propertynames(summarystats(idata; fmt="wide"))
        @test "a" ∈ s.variable
        @test "b" ∉ s.variable
        @test "b[0, 0]" ∉ s.variable
        @test "b[1, 1]" ∈ s.variable

        s2 = summarystats(idata; fmt="long")
        @test s2 isa DataFrames.DataFrame
        @test string(first(names(s2))) == "statistic"
        @test "mean" ∈ s2.statistic

        s3 = summarystats(idata; fmt="xarray")
        @test s3 isa ArviZ.Dataset
    end

    @testset "ArviZ.summary" begin
        rng = MersenneTwister(42)
        nchains, ndraws = 4, 10
        data = Dict(
            "a" => randn(rng, nchains, ndraws), "b" => randn(rng, nchains, ndraws, 3, 4)
        )

        @test ArviZ.summary(data) isa DataFrames.DataFrame
    end
end
