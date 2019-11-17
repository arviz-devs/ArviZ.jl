import Pandas

@testset "summarystats" begin
    rng = MersenneTwister(42)
    nchains, ndraws = 4, 10
    idata = convert_to_inference_data(Dict(
        "a" => randn(rng, nchains, ndraws),
        "b" => randn(rng, nchains, ndraws, 3, 4),
    ))

    @test summarystats(idata) isa Pandas.DataFrame
    @test summarystats(idata; fmt = "wide") isa Pandas.DataFrame
    @test summarystats(idata; fmt = "long") isa Pandas.DataFrame
    s = summarystats(idata)
    @test "a" ∈ Pandas.index(s)
    @test "b" ∉ Pandas.index(s)
    @test "b[0,0]" ∉ Pandas.index(s)
    @test "b[1,1]" ∈ Pandas.index(s)
    @test "b[0,0]" ∈ Pandas.index(summarystats(idata; index_origin = 0))

    s2 = summarystats(idata; fmt = "xarray")
    @test s2 isa ArviZ.Dataset
end

@testset "ArviZ.summary" begin
    rng = MersenneTwister(42)
    nchains, ndraws = 4, 10
    data = Dict(
        "a" => randn(rng, nchains, ndraws),
        "b" => randn(rng, nchains, ndraws, 3, 4),
    )

    @test ArviZ.summary(data) isa Pandas.DataFrame
end
