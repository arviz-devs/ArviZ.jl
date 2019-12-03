import DataFrames

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
    data = Dict(
        "a" => randn(rng, nchains, ndraws),
        "b" => randn(rng, nchains, ndraws, 3, 4),
    )

    @test ArviZ.summary(data) isa DataFrames.DataFrame
end
