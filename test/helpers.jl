using Random
using PyCall
using ArviZ: attributes

try
    ArviZ.initialize_bokeh()
catch
    @info "bokeh backend not available. bokeh tests will be skipped."
end

py"""
class PyNullObject(object):
   def __init__(self):
       pass
"""

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

random_dataset(args...) = ArviZ.Dataset(random_dim_stack(args...))

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

function create_model(seed=10)
    rng = MersenneTwister(seed)
    J = 8
    nchains = 4
    ndraws = 500
    data = Dict(
        "J" => J,
        "y" => [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        "sigma" => [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
    )
    posterior = Dict(
        "mu" => randn(rng, nchains, ndraws),
        "tau" => abs.(randn(rng, nchains, ndraws)),
        "eta" => randn(rng, nchains, ndraws, J),
        "theta" => randn(rng, nchains, ndraws, J),
    )
    posterior_predictive = Dict("y" => randn(rng, nchains, ndraws, J))
    sample_stats = Dict(
        "energy" => randn(rng, nchains, ndraws),
        "diverging" => (randn(rng, nchains, ndraws) .> 0.90),
        "max_depth" => (randn(rng, nchains, ndraws) .> 0.90),
        "log_likelihood" => randn(rng, nchains, ndraws, J),
    )
    prior = Dict(
        "mu" => randn(rng, nchains, ndraws) / 2,
        "tau" => abs.(randn(rng, nchains, ndraws)) / 2,
        "eta" => randn(rng, nchains, ndraws, data["J"]) / 2,
        "theta" => randn(rng, nchains, ndraws, data["J"]) / 2,
    )
    prior_predictive = Dict("y" => randn(rng, nchains, ndraws, J) / 2)
    sample_stats_prior = Dict(
        "energy" => randn(rng, nchains, ndraws),
        "diverging" => Int.(randn(rng, nchains, ndraws) .> 0.95),
    )
    model = from_dict(;
        posterior,
        posterior_predictive,
        sample_stats,
        prior,
        prior_predictive,
        sample_stats_prior,
        observed_data=Dict("y" => data["y"]),
        dims=Dict("y" => ["obs_dim"], "log_likelihood" => ["obs_dim"]),
        coords=Dict("obs_dim" => 1:J),
    )
    return model
end

models() = (model_1=create_model(10), model_2=create_model(11))

function noncentered_schools_data()
    return Dict(
        "J" => 8,
        "y" => [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        "sigma" => [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
    )
end

function check_idata_schema(idata)
    @testset "check arviz schema" begin
        @test idata isa InferenceData
        @testset for (name, group) in pairs(idata)
            @test name ∈ ArviZ.SUPPORTED_GROUPS
            @test group isa ArviZ.Dataset
            for (var_name, var_data) in pairs(group)
                @test var_data isa DimensionalData.AbstractDimArray
                if ArviZ.has_all_sample_dims(var_data)
                    @test Dimensions.name(Dimensions.dims(idata.posterior.g)[1]) === :chain
                    @test Dimensions.name(Dimensions.dims(idata.posterior.g)[2]) === :draw
                end
            end
            @testset "attributes" begin
                attrs = ArviZ.attributes(group)
                @test attrs isa AbstractDict{Symbol,Any}
                @test :created_at in keys(attrs)
                @test :arviz_version in keys(attrs)
                @test_skip :inference_library in keys(attrs)
                @test_skip :inference_library_version in keys(attrs)
                @test_skip :arviz_language in keys(attrs)
            end
        end
    end
end

function test_idata_approx_equal(
    idata1::InferenceData, idata2::InferenceData; check_metadata=true
)
    @test ArviZ.groupnames(idata1) === ArviZ.groupnames(idata2)
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

# function check_multiple_attrs(test_dict, parent)
#     failed_attrs = []
#     for (dataset_name, attributes) in test_dict
#         if Symbol(dataset_name) ∈ propertynames(parent)
#             dataset = getproperty(parent, Symbol(dataset_name))
#             for attribute in attributes
#                 if Symbol(attribute) ∉ propertynames(dataset)
#                     push!(failed_attrs, (dataset_name, attribute))
#                 end
#             end
#         else
#             push!(failed_attrs, dataset_name)
#         end
#     end
#     return failed_attrs
# end

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
    @test ArviZ.hasgroup(idata, group_name)
    ds = getproperty(idata, group_name)
    @test ds isa ArviZ.Dataset
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
    for k in [:created_at, :arviz_version]
        @test k in keys(metadata)
    end
    return nothing
end
