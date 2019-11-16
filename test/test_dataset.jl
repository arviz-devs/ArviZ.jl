@testset "dict to dataset roundtrip" begin
    rng = MersenneTwister(42)
    J = 8
    K = 6
    nchains = 4
    ndraws = 500
    vars = Dict(
        "a" => randn(rng, nchains, ndraws),
        "b" => randn(rng, nchains, ndraws, J, K),
    )
    coords = Dict("bi" => 1:J, "bj" => 1:K)
    dims = Dict("b" => ["bi", "bj"])
    attrs = Dict("mykey" => 5)

    ds = ArviZ.dict_to_dataset(vars; coords = coords, dims = dims, attrs = attrs)
    vars2, kwargs = ArviZ.dataset_to_dict(ds)
    for (k, v) in vars
        @test k ∈ keys(vars2)
        @test vars2[k] ≈ v
    end
    @test kwargs.coords == coords
    @test kwargs.dims == dims
    for (k, v) in attrs
        @test k ∈ keys(kwargs.attrs)
        @test kwargs.attrs[k] == v
    end
end
