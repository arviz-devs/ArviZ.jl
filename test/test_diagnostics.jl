@testset "diagnostics" begin
    @testset "bfmi" begin
        idata = load_arviz_data("centered_eight")
        @test bfmi(idata) == ArviZ.arviz.bfmi(idata)
        rng = Random.MersenneTwister(42)
        arr = randn(rng, 4, 100)
        @test bfmi(arr) == ArviZ.arviz.bfmi(arr)
    end

    @testset "ess" begin
        rng = Random.MersenneTwister(42)
        arr = randn(rng, 4, 100)
        @test ess(arr) == ArviZ.arviz.ess(arr)
        @test ess((x=arr,)) isa ArviZ.Dataset
        @test ess((x=arr,)).x.values == ArviZ.arviz.ess(Dict(:x=>arr)).x.values
    end

    @testset "mcse" begin
        rng = Random.MersenneTwister(42)
        arr = randn(rng, 4, 100)
        @test mcse(arr) == ArviZ.arviz.mcse(arr)
        @test mcse((x=arr,)) isa ArviZ.Dataset
        @test mcse((x=arr,)).x.values == ArviZ.arviz.mcse(Dict(:x=>arr)).x.values
    end

    @testset "rhat" begin
        rng = Random.MersenneTwister(42)
        arr = randn(rng, 4, 100)
        @test rhat(arr) == ArviZ.arviz.rhat(arr)
        @test rhat((x=arr,)) isa ArviZ.Dataset
        @test rhat((x=arr,)).x.values == ArviZ.arviz.rhat(Dict(:x=>arr)).x.values
    end

    @testset "geweke" begin
        rng = Random.MersenneTwister(42)
        arr = randn(rng, 1000)
        @test geweke(arr, 0.1, 0.5, 10) == ArviZ.arviz.geweke(arr, 0.1, 0.5, 10)
    end
end
