import DataFrames
import Pandas

@testset "stats" begin
    idata = load_arviz_data("centered_eight")

    @testset "compare" begin
        idata2 = load_arviz_data("non_centered_eight")
        df = compare(Dict("a"=>idata, "b"=>idata2))
        @test df isa DataFrames.DataFrame
    end

    @testset "hpd" begin
        rng = Random.MersenneTwister(42)
        x = randn(rng, 100)
        @test hpd(x) == ArviZ.arviz.hpd(x)
    end

    @testset "r2_score" begin
        rng = Random.MersenneTwister(42)
        ytrue = randn(rng, 100)
        ypred = randn(rng, 100)
        df = r2_score(ytrue, ypred)
        @test df isa DataFrames.DataFrame
        @test all(df == ArviZ.todataframes(Pandas.Series(ArviZ.arviz.r2_score(ytrue, ypred))))
    end

    @testset "loo" begin
        df = loo(idata)
        @test df isa DataFrames.DataFrame
        @test all(df == ArviZ.todataframes(Pandas.Series(ArviZ.arviz.loo(idata))))
    end

    @testset "waic" begin
        df = waic(idata)
        @test df isa DataFrames.DataFrame
        @test all(df == ArviZ.todataframes(Pandas.Series(ArviZ.arviz.waic(idata))))
    end

    @testset "loo_pit" begin
        ret = loo_pit(idata; y = "obs")
        @test ret == ArviZ.arviz.loo_pit(idata; y = "obs")
    end

    @testset "summarystats" begin
        rng = MersenneTwister(42)
        nchains, ndraws = 4, 10
        idata = convert_to_inference_data(Dict(
            "a" => randn(rng, nchains, ndraws),
            "b" => randn(rng, nchains, ndraws, 3, 4),
        ))

        @test summarystats(idata) isa DataFrames.DataFrame
        @test summarystats(idata; fmt = "wide") isa DataFrames.DataFrame
        @test summarystats(idata; fmt = "long") isa DataFrames.DataFrame
        s = summarystats(idata)
        @test :variable in propertynames(s)
        @test "a" ∈ s.variable
        @test "b" ∉ s.variable
        @test "b[0,0]" ∉ s.variable
        @test "b[1,1]" ∈ s.variable
        @test "b[0,0]" ∈ summarystats(idata; index_origin = 0).variable

        s2 = summarystats(idata; fmt = "xarray")
        @test s2 isa ArviZ.Dataset
    end

    @testset "ArviZ.summary" begin
        rng = MersenneTwister(42)
        nchains, ndraws = 4, 10
        data =
            Dict("a" => randn(rng, nchains, ndraws), "b" => randn(rng, nchains, ndraws, 3, 4))

        @test ArviZ.summary(data) isa DataFrames.DataFrame
    end
end
