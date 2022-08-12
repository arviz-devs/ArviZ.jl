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
