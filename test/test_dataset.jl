@testset "ArviZ.Dataset" begin
    data = load_arviz_data("centered_eight")
    dataset = data.posterior
    @test dataset isa ArviZ.Dataset

    @testset "construction" begin
        pydataset = PyObject(dataset)
        @test ArviZ.Dataset(pydataset) isa ArviZ.Dataset
        @test PyObject(ArviZ.Dataset(pydataset)) === pydataset
        @test ArviZ.Dataset(dataset) === dataset
    end

    @testset "properties" begin
        @test length(propertynames(dataset)) > 1
    end

    @testset "conversion" begin
        @test pyisinstance(PyObject(dataset), ArviZ.xarray.Dataset)
        dataset4 = convert(ArviZ.Dataset, PyObject(dataset))
        @test dataset4 isa ArviZ.Dataset
        @test PyObject(dataset4) === PyObject(dataset)
    end
end

@testset "ArviZ.convert_to_dataset(data::ArviZ.Dataset; kwargs...)" begin
    data = load_arviz_data("centered_eight")
    dataset = data.posterior
    @test ArviZ.convert_to_dataset(dataset) isa ArviZ.Dataset
    @test ArviZ.convert_to_dataset(dataset) === dataset
end

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
    @test ds isa ArviZ.Dataset
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
