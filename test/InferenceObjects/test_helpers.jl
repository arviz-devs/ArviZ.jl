using ArviZ.InferenceObjects, DimensionalData

function random_dim_array(var_name, dims, coords, default_dims=())
    _dims = (default_dims..., dims...)
    _coords = NamedTuple{_dims}(getproperty.(Ref(coords), _dims))
    size = map(length, values(_coords))
    data = randn(size)
    return DimArray(data, _coords; name=var_name)
end

function random_dim_stack(var_names, dims, coords, metadata, default_dims=(:chain, :draw))
    dim_arrays = map(var_names) do k
        return random_dim_array(k, getproperty(dims, k), coords, default_dims)
    end
    return DimStack(dim_arrays...; metadata)
end

random_dataset(args...) = Dataset(random_dim_stack(args...))

function random_data()
    var_names = (:a, :b)
    data_names = (:y,)
    coords = (
        chain=1:4, draw=1:100, shared=["s1", "s2", "s3"], dima=1:4, dimb=2:6, dimy=1:5
    )
    dims = (a=(:shared, :dima), b=(:shared, :dimb), y=(:shared, :dimy))
    metadata = (inference_library="PPL",)
    posterior = random_dataset(var_names, dims, coords, metadata)
    posterior_predictive = random_dataset(data_names, dims, coords, metadata)
    prior = random_dataset(var_names, dims, coords, metadata)
    prior_predictive = random_dataset(data_names, dims, coords, metadata)
    observed_data = random_dataset(data_names, dims, coords, metadata, ())
    return InferenceData(;
        posterior, posterior_predictive, prior, prior_predictive, observed_data
    )
end

function check_idata_schema(idata)
    @testset "check InferenceData schema" begin
        @test idata isa InferenceData
        @testset "$name" for (name, group) in pairs(idata)
            @test name ∈ InferenceObjects.SCHEMA_GROUPS
            @test group isa Dataset
            for (var_name, var_data) in pairs(group)
                @test var_data isa DimensionalData.AbstractDimArray
                if InferenceObjects.has_all_sample_dims(var_data)
                    @test Dimensions.name(Dimensions.dims(var_data)[1]) === :chain
                    @test Dimensions.name(Dimensions.dims(var_data)[2]) === :draw
                end
            end
            @testset "attributes" begin
                attrs = InferenceObjects.attributes(group)
                @test attrs isa AbstractDict{Symbol,Any}
                @test :created_at in keys(attrs)
            end
        end
    end
end

function test_idata_approx_equal(
    idata1::InferenceData, idata2::InferenceData; check_metadata=true
)
    @test InferenceObjects.groupnames(idata1) === InferenceObjects.groupnames(idata2)
    for (ds1, ds2) in zip(idata1, idata2)
        @test issetequal(keys(ds1), keys(ds2))
        for var_name in keys(ds1)
            da1 = ds1[var_name]
            da2 = ds2[var_name]
            @test da1 ≈ da2
            dims1 = DimensionalData.dims(da1)
            dims2 = DimensionalData.dims(da2)
            @test DimensionalData.name(dims1) == DimensionalData.name(dims2)
            @test DimensionalData.index(dims1) == DimensionalData.index(dims2)
        end
        if check_metadata
            metadata1 = DimensionalData.metadata(ds1)
            metadata2 = DimensionalData.metadata(ds2)
            @test issetequal(keys(metadata1), keys(metadata2))
            for k in keys(metadata1)
                Symbol(k) === :created_at && continue
                @test metadata1[k] == metadata2[k]
            end
        end
    end
end

function test_idata_group_correct(
    idata,
    group_name,
    var_names;
    library=nothing,
    dims=(;),
    coords=(;),
    default_dims=(:chain, :draw),
)
    @test idata isa InferenceData
    @test InferenceObjects.hasgroup(idata, group_name)
    ds = getproperty(idata, group_name)
    @test ds isa Dataset
    @test issetequal(keys(ds), var_names)
    for name in var_names
        da = ds[name]
        @test DimensionalData.name(da) === name
        _dims = DimensionalData.dims(da)
        _dim_names_exp = (default_dims..., get(dims, name, ())...)
        _dim_names = DimensionalData.name(_dims)
        @test issubset(_dim_names_exp, _dim_names)
        for dim in _dims
            dim_name = DimensionalData.name(dim)
            if dim_name ∈ keys(coords)
                @test coords[dim_name] == DimensionalData.index(dim)
            end
        end
    end
    metadata = DimensionalData.metadata(ds)
    if library !== nothing
        @test metadata[:inference_library] == library
    end
    for k in [:created_at]
        @test k in keys(metadata)
    end
    return nothing
end
