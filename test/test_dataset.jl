using ArviZ, DimensionalData, OrderedCollections, PyCall, Test

@testset "dataset" begin
    @testset "ArviZ.Dataset" begin
        @testset "Constructors" begin
            nchains = 4
            ndraws = 100
            nshared = 3
            xdims = (:chain, :draw, :shared)
            x = DimArray(randn(nchains, ndraws, nshared), xdims)
            ydims = (:chain, :draw, :ydim1, :shared)
            y = DimArray(randn(nchains, ndraws, 2, nshared), ydims)
            metadata = Dict(:prop1 => "val1", :prop2 => "val2")

            @testset "from NamedTuple" begin
                data = (; x, y)
                ds = ArviZ.Dataset(data; metadata)
                @test ds isa ArviZ.Dataset
                @test DimensionalData.data(ds) == data
                for dim in xdims
                    @test DimensionalData.hasdim(ds, dim)
                end
                for dim in ydims
                    @test DimensionalData.hasdim(ds, dim)
                end
                for (var_name, dims) in ((:x, xdims), (:y, ydims))
                    da = ds[var_name]
                    @test DimensionalData.name(da) === var_name
                    @test DimensionalData.name(DimensionalData.dims(da)) === dims
                end
                @test DimensionalData.metadata(ds) == metadata
            end

            @testset "from DimArrays" begin
                data = (
                    DimensionalData.rebuild(x; name=:x), DimensionalData.rebuild(y; name=:y)
                )
                ds = ArviZ.Dataset(data...; metadata)
                @test ds isa ArviZ.Dataset
                @test values(DimensionalData.data(ds)) == data
                for dim in xdims
                    @test DimensionalData.hasdim(ds, dim)
                end
                for dim in ydims
                    @test DimensionalData.hasdim(ds, dim)
                end
                for (var_name, dims) in ((:x, xdims), (:y, ydims))
                    da = ds[var_name]
                    @test DimensionalData.name(da) === var_name
                    @test DimensionalData.name(DimensionalData.dims(da)) === dims
                end
                @test DimensionalData.metadata(ds) == metadata
            end

            @testset "errors with mismatched dimensions" begin
                data_bad = (
                    x=DimArray(randn(3, 100, 3), (:chains, :draws, :shared)),
                    y=DimArray(randn(4, 100, 2, 3), (:chains, :draws, :ydim1, :shared)),
                )
                @test_throws Exception ArviZ.Dataset(data_bad)
            end
        end

        nchains = 4
        ndraws = 100
        nshared = 3
        xdims = (:chain, :draw, :shared)
        x = DimArray(randn(nchains, ndraws, nshared), xdims)
        ydims = (:chain, :draw, :ydim1, :shared)
        y = DimArray(randn(nchains, ndraws, 2, nshared), ydims)
        metadata = Dict(:prop1 => "val1", :prop2 => "val2")
        ds = ArviZ.Dataset((; x, y); metadata)

        @testset "parent" begin
            @test parent(ds) isa DimStack
            @test parent(ds) == ds
        end

        @testset "properties" begin
            @test propertynames(ds) == (:x, :y)
            @test ds.x isa DimArray
            @test ds.x == x
            @test ds.y isa DimArray
            @test ds.y == y
        end

        @testset "getindex" begin
            @test ds[:x] isa DimArray
            @test ds[:x] == x
            @test ds[:y] isa DimArray
            @test ds[:y] == y
        end

        @testset "copy/deepcopy" begin
            @test copy(ds) == ds
            @test deepcopy(ds) == ds
        end

        @testset "attributes" begin
            @test ArviZ.attributes(ds) == metadata
            dscopy = deepcopy(ds)
            ArviZ.setattribute!(dscopy, :prop3, "val3")
            @test ArviZ.attributes(dscopy)[:prop3] == "val3"
            @test_deprecated ArviZ.setattribute!(dscopy, "prop3", "val4")
            @test ArviZ.attributes(dscopy)[:prop3] == "val4"
        end

        @testset "conversion" begin
            @test convert(ArviZ.Dataset, ds) === ds
            ds2 = convert(ArviZ.Dataset, [1.0, 2.0, 3.0, 4.0])
            @test ds2 isa ArviZ.Dataset
            @test ds2 == ArviZ.convert_to_dataset([1.0, 2.0, 3.0, 4.0])
        end
    end

    @testset "Dataset <-> xarray" begin
        nchains = 4
        ndraws = 100
        nshared = 3
        xdims = (:chain, :draw, :shared)
        x = DimArray(randn(nchains, ndraws, nshared), xdims)
        ydims = (:chain, :draw, Dim{:ydim1}(Any["a", "b"]), :shared)
        y = DimArray(randn(nchains, ndraws, 2, nshared), ydims)
        metadata = Dict(:prop1 => "val1", :prop2 => "val2")
        ds = ArviZ.Dataset((; x, y); metadata)
        o = PyObject(ds)
        @test o isa PyObject
        @test pyisinstance(o, ArviZ.xarray.Dataset)

        @test issetequal(Symbol.(o.coords.keys()), (:chain, :draw, :shared, :ydim1))
        for (dim, coord) in o.coords.items()
            @test collect(coord.values) == DimensionalData.index(ds, Symbol(dim))
        end

        variables = Dict(collect(o.data_vars.variables.items()))
        @test "x" ∈ keys(variables)
        @test x == variables["x"].values
        @test variables["x"].dims == String.(xdims)

        @test "y" ∈ keys(variables)
        @test y == variables["y"].values
        @test variables["y"].dims == ("chain", "draw", "ydim1", "shared")

        # check that the Python object accesses the underlying Julia array
        x[1] = 1
        @test x == variables["x"].values

        ds2 = convert(ArviZ.Dataset, o)
        @test ds2 isa ArviZ.Dataset
        @test ds2.x ≈ ds.x
        @test ds2.y ≈ ds.y
        dims1 = sort(collect(DimensionalData.dims(ds)); by=DimensionalData.name)
        dims2 = sort(collect(DimensionalData.dims(ds2)); by=DimensionalData.name)
        for (dim1, dim2) in zip(dims1, dims2)
            @test DimensionalData.name(dim1) === DimensionalData.name(dim2)
            @test DimensionalData.index(dim1) == DimensionalData.index(dim2)
            if DimensionalData.index(dim1) isa AbstractRange
                @test DimensionalData.index(dim2) isa AbstractRange
            end
        end
        @test DimensionalData.metadata(ds2) == DimensionalData.metadata(ds)
    end

    @testset "namedtuple_to_dataset" begin
        J = 8
        K = 6
        L = 3
        nchains = 4
        ndraws = 500
        vars = (a=randn(nchains, ndraws, J), b=randn(nchains, ndraws, K, L))
        coords = (bi=2:(K + 1), draw=1:2:1_000)
        dims = (b=[:bi, nothing],)
        expected_dims = (
            a=(
                Dimensions.Dim{:chain}(1:nchains),
                Dimensions.Dim{:draw}(1:2:1_000),
                Dimensions.Dim{:a_dim_1}(1:J),
            ),
            b=(
                Dimensions.Dim{:chain}(1:nchains),
                Dimensions.Dim{:draw}(1:2:1_000),
                Dimensions.Dim{:bi}(2:(K + 1)),
                Dimensions.Dim{:b_dim_2}(1:L),
            ),
        )
        attrs = Dict(:mykey => 5)
        @test_broken @inferred ArviZ.namedtuple_to_dataset(
            vars; library="MyLib", coords, dims, attrs
        )
        ds = ArviZ.namedtuple_to_dataset(vars; library="MyLib", coords, dims, attrs)
        @test ds isa ArviZ.Dataset
        for (var_name, var_data) in pairs(DimensionalData.layers(ds))
            @test var_data isa DimensionalData.DimArray
            @test var_name === DimensionalData.name(var_data)
            @test var_data == vars[var_name]
            _dims = DimensionalData.dims(var_data)
            @test _dims == expected_dims[var_name]
        end
        metadata = DimensionalData.metadata(ds)
        @test metadata isa OrderedDict
        @test haskey(metadata, :created_at)
        @test haskey(metadata, :arviz_version)
        @test metadata[:arviz_language] == "julia"
        @test metadata[:inference_library] == "MyLib"
        @test !haskey(metadata, :inference_library_version)
        @test metadata[:mykey] == 5
    end

    @testset "ArviZ.convert_to_dataset" begin
        nchains = 4
        ndraws = 100
        nshared = 3
        xdims = (:chain, :draw, :shared)
        x = DimArray(randn(nchains, ndraws, nshared), xdims)
        ydims = (:chain, :draw, Dim{:ydim1}(Any["a", "b"]), :shared)
        y = DimArray(randn(nchains, ndraws, 2, nshared), ydims)
        metadata = Dict(:prop1 => "val1", :prop2 => "val2")
        ds = ArviZ.Dataset((; x, y); metadata)

        @testset "ArviZ.convert_to_dataset(::ArviZ.Dataset; kwargs...)" begin
            @test ArviZ.convert_to_dataset(ds) isa ArviZ.Dataset
            @test ArviZ.convert_to_dataset(ds) === ds
        end

        @testset "ArviZ.convert_to_dataset(::$T; kwargs...)" for T in (Dict, NamedTuple)
            data = (x=randn(4, 100), y=randn(4, 100, 2))
            if T <: Dict
                data = T(pairs(data))
            end
            ds2 = ArviZ.convert_to_dataset(data)
            @test ds2 isa ArviZ.Dataset
            @test ds2.x == data[:x]
            @test DimensionalData.name(DimensionalData.dims(ds2.x)) == (:chain, :draw)
            @test ds2.y == data[:y]
            @test DimensionalData.name(DimensionalData.dims(ds2.y)) ==
                (:chain, :draw, :y_dim_1)
        end
    end

    @testset "ArviZ.convert_to_constant_dataset" begin
        @testset "ArviZ.convert_to_constant_dataset(::Dict)" begin
            data = Dict(:x => randn(4, 5), :y => ["a", "b", "c"])
            ds = ArviZ.convert_to_constant_dataset(data)
            @test ds isa ArviZ.Dataset
            @test issetequal(keys(ds), keys(data))
            @test :x ∈ keys(ds)
            @test :y ∈ keys(ds)
            @test issetequal(
                DimensionalData.name(DimensionalData.dims(ds)),
                (:x_dim_0, :x_dim_1, :y_dim_0),
            )
            @test ds.x == data[:x]
            @test ds.y == data[:y]
        end

        @testset "ArviZ.convert_to_constant_dataset(::Dict; kwargs...)" begin
            data = Dict(:x => randn(4, 5), :y => ["a", "b", "c"])
            coords = Dict(:xdim1 => 1:4, :xdim2 => 5:9, :ydim1 => ["d", "e", "f"])
            dims = Dict(:x => [:xdim1, :xdim2], :y => [:ydim1])
            library = "MyLib"
            attrs = Dict(:prop => "propval")
            ds = ArviZ.convert_to_constant_dataset(data; coords, dims, library, attrs)
            @test ds isa ArviZ.Dataset
            @test :x ∈ keys(ds)
            @test :y ∈ keys(ds)
            @test issetequal(DimensionalData.name(DimensionalData.dims(ds)), keys(coords))
            @test ds.x == data[:x]
            @test ds.y == data[:y]
            for var_name in keys(dims)
                @test ds[var_name] == data[var_name]
                @test collect(DimensionalData.name(DimensionalData.dims(ds[var_name]))) ==
                    dims[var_name]
            end
            for k in keys(coords)
                @test DimensionalData.index(ds, k) == coords[k]
            end
            @test ArviZ.attributes(ds)[:prop] == "propval"
            @test ArviZ.attributes(ds)[:inference_library] == library
        end

        @testset "ArviZ.convert_to_constant_dataset(::NamedTuple; kwargs...)" begin
            data = (x=randn(4, 5), y=["a", "b", "c"])
            coords = (xdim1=1:4, xdim2=5:9, ydim1=["d", "e", "f"])
            dims = (x=[:xdim1, :xdim2], y=[:ydim1])
            library = "MyLib"
            dataset = ArviZ.convert_to_constant_dataset(data)
            attrs = (prop="propval",)

            ds = ArviZ.convert_to_constant_dataset(data; coords, dims, library, attrs)
            @test ds isa ArviZ.Dataset
            @test :x ∈ keys(ds)
            @test :y ∈ keys(ds)
            @test issetequal(DimensionalData.name(DimensionalData.dims(ds)), keys(coords))
            @test ds.x == data.x
            @test ds.y == data.y
            for var_name in keys(dims)
                @test ds[var_name] == getproperty(data, var_name)
                @test collect(DimensionalData.name(DimensionalData.dims(ds[var_name]))) ==
                    getproperty(dims, var_name)
            end
            for k in keys(coords)
                @test DimensionalData.index(ds, k) == getproperty(coords, k)
            end
            @test ArviZ.attributes(ds)[:prop] == "propval"
            @test ArviZ.attributes(ds)[:inference_library] == library
        end
    end

    @testset "dict to dataset roundtrip" begin
        J = 8
        K = 6
        nchains = 4
        ndraws = 500
        vars = Dict(:a => randn(nchains, ndraws), :b => randn(nchains, ndraws, J, K))
        coords = Dict(:bi => 1:J, :bj => 1:K)
        dims = Dict(:b => [:bi, :bj])
        attrs = Dict(:mykey => 5)

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
        @test :inference_library ∈ keys(kwargs.attrs)
        @test kwargs.attrs[:inference_library] == "MyLib"
    end
end
