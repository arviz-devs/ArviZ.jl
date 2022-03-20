@testset "ArviZ.Dataset" begin
    data = load_arviz_data("centered_eight")
    dataset = data.posterior
    @test dataset isa ArviZ.Dataset

    @testset "construction" begin
        pydataset = PyObject(dataset)
        @test ArviZ.Dataset(pydataset) isa ArviZ.Dataset
        @test PyObject(ArviZ.Dataset(pydataset)) === pydataset
        @test ArviZ.Dataset(dataset) === dataset
        @test_throws ArgumentError ArviZ.Dataset(py"PyNullObject()")
        @test hash(dataset) == hash(pydataset)

        vars = Dict("x" => ("dimx", randn(3)), ("y" => (("dimy_1", "dimy_2"), randn(3, 2))))
        coords = Dict(
            "dimx" => [1, 2, 3], "dimy_1" => ["a", "b", "c"], "dimy_2" => ["d", "e"]
        )
        attrs = Dict("prop1" => 1, "prop2" => "propval")
        @inferred ArviZ.Dataset(; data_vars=vars, coords, attrs)
        ds = ArviZ.Dataset(; data_vars=vars, coords, attrs)
        @test ds isa ArviZ.Dataset
        vars2, kwargs = ArviZ.dataset_to_dict(ds)
        for (k, v) in vars
            @test k ∈ keys(vars2)
            @test vars2[k] ≈ v[2]
        end
        @test kwargs.coords == coords
        for (k, v) in attrs
            @test k ∈ keys(kwargs.attrs)
            @test kwargs.attrs[k] == v
        end
    end

    @testset "properties" begin
        @test length(propertynames(dataset)) > 1
        @test dataset["mu"] isa PyObject
        @test dataset["mu"].values == py"$(dataset)['mu'].values"
    end

    @testset "conversion" begin
        @test pyisinstance(PyObject(dataset), ArviZ.xarray.Dataset)
        dataset4 = convert(ArviZ.Dataset, PyObject(dataset))
        @test dataset4 isa ArviZ.Dataset
        @test PyObject(dataset4) === PyObject(dataset)

        # TODO: improve this test
        @test convert(ArviZ.Dataset, [1.0, 2.0, 3.0, 4.0]) isa ArviZ.Dataset
    end

    @testset "show(::ArviZ.Dataset)" begin
        @testset "$mimetype" for mimetype in ("plain", "html")
            text = repr(MIME("text/$(mimetype)"), dataset)
            @test text isa String
            @test !occursin("<xarray.Dataset>", text)
            @test !occursin("&lt;xarray.Dataset&gt;", text)
            @test occursin("Dataset (xarray.Dataset)", text)
        end
    end
end

@testset "ArviZ.convert_to_dataset" begin
    rng = MersenneTwister(42)

    @testset "ArviZ.convert_to_dataset(::ArviZ.Dataset; kwargs...)" begin
        data = load_arviz_data("centered_eight")
        dataset = data.posterior
        @test ArviZ.convert_to_dataset(dataset) isa ArviZ.Dataset
        @test ArviZ.convert_to_dataset(dataset) === dataset
    end

    @testset "ArviZ.convert_to_dataset(::InferenceData; kwargs...)" begin
        A = Dict("A" => randn(rng, 2, 10, 2))
        B = Dict("B" => randn(rng, 2, 10, 2))
        dataA = ArviZ.convert_to_dataset(A)
        dataB = ArviZ.convert_to_dataset(B)
        idata = InferenceData(; posterior=dataA, prior=dataB)

        ds1 = ArviZ.convert_to_dataset(idata)
        @test ds1 isa ArviZ.Dataset
        @test "A" ∈ [ds1.keys()...]

        ds2 = ArviZ.convert_to_dataset(idata; group=:prior)
        @test ds2 isa ArviZ.Dataset
        @test "B" ∈ [ds2.keys()...]
    end

    @testset "ArviZ.convert_to_dataset(::Dict; kwargs...)" begin
        data = Dict("x" => randn(rng, 4, 100), "y" => randn(rng, 4, 100))
        dataset = ArviZ.dict_to_dataset(data)
        @test dataset isa ArviZ.Dataset
        @test "x" ∈ [dataset.keys()...]
        @test "y" ∈ [dataset.keys()...]
    end
end

@testset "ArviZ.convert_to_constant_dataset" begin
    @testset "ArviZ.convert_to_constant_dataset(::Dict)" begin
        data = Dict("x" => randn(4, 5), "y" => ["a", "b", "c"])
        dataset = ArviZ.convert_to_constant_dataset(data)
        @test dataset isa ArviZ.Dataset
        @test "x" ∈ dataset.keys()
        @test "y" ∈ dataset.keys()
        @test Set(dataset.coords) == Set(["x_dim_0", "x_dim_1", "y_dim_0"])
        @test collect(dataset._variables["x"].values) == data["x"]
        @test collect(dataset._variables["y"].values) == data["y"]
    end

    @testset "ArviZ.convert_to_constant_dataset(::Dict; kwargs...)" begin
        data = Dict("x" => randn(4, 5), "y" => ["a", "b", "c"])
        coords = Dict("xdim1" => 1:4, "xdim2" => 5:9, "ydim1" => ["d", "e", "f"])
        dims = Dict("x" => ["xdim1", "xdim2"], "y" => ["ydim1"])
        library = "MyLib"
        dataset = ArviZ.convert_to_constant_dataset(data)
        attrs = Dict("prop" => "propval")

        dataset = ArviZ.convert_to_constant_dataset(data; coords, dims, library, attrs)
        @test dataset isa ArviZ.Dataset
        @test "x" ∈ dataset.keys()
        @test "y" ∈ dataset.keys()
        @test Set(dataset.coords) == Set(["xdim1", "xdim2", "ydim1"])
        @test collect(dataset._variables["xdim1"].values) == coords["xdim1"]
        @test collect(dataset._variables["xdim2"].values) == coords["xdim2"]
        @test collect(dataset._variables["ydim1"].values) == coords["ydim1"]
        @test collect(dataset["x"].coords) == ["xdim1", "xdim2"]
        @test collect(dataset["y"].coords) == ["ydim1"]
        @test collect(dataset._variables["x"].values) == data["x"]
        @test collect(dataset._variables["y"].values) == data["y"]
        @test dataset.attrs["prop"] == attrs["prop"]
        @test dataset.attrs["inference_library"] == library
    end

    @testset "ArviZ.convert_to_constant_dataset(::NamedTuple; kwargs...)" begin
        data = (x=randn(4, 5), y=["a", "b", "c"])
        coords = (xdim1=1:4, xdim2=5:9, ydim1=["d", "e", "f"])
        dims = (x=["xdim1", "xdim2"], y=["ydim1"])
        library = "MyLib"
        dataset = ArviZ.convert_to_constant_dataset(data)
        attrs = (prop="propval",)

        dataset = ArviZ.convert_to_constant_dataset(data; coords, dims, library, attrs)
        @test dataset isa ArviZ.Dataset
        @test "x" ∈ dataset.keys()
        @test "y" ∈ dataset.keys()
        @test Set(dataset.coords) == Set(["xdim1", "xdim2", "ydim1"])
        @test collect(dataset._variables["xdim1"].values) == coords.xdim1
        @test collect(dataset._variables["xdim2"].values) == coords.xdim2
        @test collect(dataset._variables["ydim1"].values) == coords.ydim1
        @test collect(dataset["x"].coords) == ["xdim1", "xdim2"]
        @test collect(dataset["y"].coords) == ["ydim1"]
        @test collect(dataset._variables["x"].values) == data.x
        @test collect(dataset._variables["y"].values) == data.y
        @test dataset.attrs["prop"] == attrs.prop
        @test dataset.attrs["inference_library"] == library
    end
end

@testset "dict to dataset roundtrip" begin
    rng = MersenneTwister(42)
    J = 8
    K = 6
    nchains = 4
    ndraws = 500
    vars = Dict(
        "a" => randn(rng, nchains, ndraws), "b" => randn(rng, nchains, ndraws, J, K)
    )
    coords = Dict("bi" => 1:J, "bj" => 1:K)
    dims = Dict("b" => ["bi", "bj"])
    attrs = Dict("mykey" => 5)

    ds = ArviZ.dict_to_dataset(vars; library=:MyLib, coords, dims, attrs)
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
    @test "inference_library" ∈ keys(kwargs.attrs)
    @test kwargs.attrs["inference_library"] == "MyLib"
end
