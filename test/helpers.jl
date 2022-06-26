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
    return InferenceData(; posterior, posterior_predictive, prior, prior_predictive, observed_data)
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

function check_multiple_attrs(test_dict, parent)
    failed_attrs = []
    for (dataset_name, attributes) in test_dict
        if Symbol(dataset_name) ∈ propertynames(parent)
            dataset = getproperty(parent, Symbol(dataset_name))
            for attribute in attributes
                if Symbol(attribute) ∉ propertynames(dataset)
                    push!(failed_attrs, (dataset_name, attribute))
                end
            end
        else
            push!(failed_attrs, dataset_name)
        end
    end
    return failed_attrs
end

dimsizes(ds) = ds._dims
convertindex(x::AbstractArray) = x
convertindex(o::PyObject) = o.array.values
vardict(ds) = Dict(k => convertindex(v._data) for (k, v) in ds._variables)
dimdict(ds) = Dict(k => v._dims for (k, v) in ds._variables)

function test_namedtuple_data(
    idata, group, names, nchains, ndraws; library="MyLib", coords=Dict(), dims=Dict()
)
    @test idata isa InferenceData
    @test group in ArviZ.groupnames(idata)
    ds = getproperty(idata, group)
    sizes = dimsizes(ds)
    @test length(sizes) == 2 + length(coords)
    vars = vardict(ds)
    for name in string.(names)
        @test name in keys(vars)
        dim = get(dims, name, get(dims, Symbol(name), []))
        s = (x -> length(get(coords, x, get(coords, Symbol(x), [])))).(dim)
        @test size(vars[name]) == (nchains, ndraws, s...)
    end
    @test "inference_library" in keys(attributes(ds))
    @test attributes(ds)["inference_library"] == library
    return nothing
end
