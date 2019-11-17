import Pandas

@testset "summarize" begin
    rng = MersenneTwister(42)
    nchains, ndraws = 4, 10
    idata = convert_to_inference_data(Dict(
        "a" => randn(rng, nchains, ndraws),
        "b" => randn(rng, nchains, ndraws, 3, 4),
    ))

    @test summarize(idata) isa Pandas.DataFrame
    @test summarize(idata; fmt = "wide") isa Pandas.DataFrame
    @test summarize(idata; fmt = "long") isa Pandas.DataFrame
    s = summarize(idata)
    @test "a" ∈ Pandas.index(s)
    @test "b" ∉ Pandas.index(s)
    @test "b[0,0]" ∉ Pandas.index(s)
    @test "b[1,1]" ∈ Pandas.index(s)
    @test "b[0,0]" ∈ Pandas.index(summarize(idata; index_origin = 0))

    s2 = summarize(idata; fmt = "xarray")
    @test s2 isa ArviZ.Dataset
end

@testset "summary" begin
    rng = MersenneTwister(42)
    nchains, ndraws = 4, 10
    idata = convert_to_inference_data(Dict(
        "a" => randn(rng, nchains, ndraws),
        "b" => randn(rng, nchains, ndraws, 3, 4),
    ))

    @test summary(idata) isa Pandas.DataFrame
end
