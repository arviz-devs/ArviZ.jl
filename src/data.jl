"""
    InferenceData(::PyObject)
    InferenceData(; kwargs...)

Loose wrapper around `arviz.InferenceData`, which is a container for inference
data storage using xarray.

`InferenceData` can be constructed either from an `arviz.InferenceData`
or from multiple [`Dataset`](@ref)s assigned to groups specified as `kwargs`.

Instead of directly creating an `InferenceData`, use the exported `from_xyz`
functions or [`convert_to_inference_data`](@ref).
"""
struct InferenceData
    o::PyObject

    function InferenceData(o::PyObject)
        pyisinstance(
            o,
            arviz.InferenceData,
        ) || throw(ArgumentError("$o is not an `arviz.InferenceData`."))
        return new(o)
    end
end

InferenceData(; kwargs...) = arviz.InferenceData(; kwargs...)

@inline InferenceData(data::InferenceData) = data

@inline PyObject(data::InferenceData) = getfield(data, :o)

Base.convert(::Type{InferenceData}, o::PyObject) = InferenceData(o)

Base.hash(data::InferenceData) = hash(PyObject(data))

Base.propertynames(data::InferenceData) = propertynames(PyObject(data))

function Base.getproperty(data::InferenceData, name::Symbol)
    o = PyObject(data)
    name === :o && return o
    return getproperty(o, name)
end

Base.delete!(data::InferenceData, name) = PyObject(data).__delattr__(string(name))

function (data1::InferenceData + data2::InferenceData)
    return InferenceData(PyObject(data1) + PyObject(data2))
end

function Base.show(io::IO, data::InferenceData)
    out = pycall(pybuiltin("str"), String, data)
    out = replace(out, "Inference data" => "InferenceData")
    print(io, out)
end

"""
    groupnames(data::InferenceData) -> Vector{Symbol}

Get the names of the groups (datasets) in `data`.
"""
groupnames(data::InferenceData) = Symbol.(PyObject(data)._groups)

"""
    groups(data::InferenceData) -> Dict{Symbol,Dataset}

Get the groups in `data` as a dictionary mapping names to datasets.
"""
groups(data::InferenceData) =
    Dict((name => getproperty(data, name) for name in groupnames(data)))

Base.isempty(data::InferenceData) = isempty(groupnames(data))

@forwardfun convert_to_inference_data

convert_to_inference_data(::Nothing; kwargs...) = InferenceData()

function convert_to_dataset(data::InferenceData; group = :posterior, kwargs...)
    group = Symbol(group)
    dataset = getproperty(data, group)
    return dataset
end

@forwardfun load_arviz_data

@forwardfun to_netcdf
@forwardfun from_netcdf
@forwardfun from_dict
@forwardfun from_cmdstan
@forwardfun from_cmdstanpy
@forwardfun from_emcee
@forwardfun from_pymc3
@forwardfun from_pyro
@forwardfun from_numpyro
@forwardfun from_pystan
@forwardfun from_tfp

# A more flexible form of `from_dict`
# Internally calls `dict_to_dataset`
function _from_dict(
    posterior = nothing;
    attrs = Dict(),
    coords = nothing,
    dims = nothing,
    dicts...,
)
    dicts = (posterior = posterior, dicts...)

    datasets = []
    for (name, dict) in pairs(dicts)
        (dict === nothing || isempty(dict)) && continue
        dataset = dict_to_dataset(dict; attrs = attrs, coords = coords, dims = dims)
        push!(datasets, name => dataset)
    end

    idata = InferenceData(; datasets...)
    return idata
end

@doc forwarddoc(:concat) function concat(data::InferenceData...; kwargs...)
    data = Iterators.filter(x -> !isempty(x), data)
    return arviz.concat(data...; inplace = false, kwargs...)
end

Docs.getdoc(::typeof(concat)) = forwardgetdoc(:concat)

"""
    concat!(data1::InferenceData, data::InferenceData...; kwargs...) -> InferenceData

In-place version of `concat`, where `data1` is modified to contain the
concatenation of `data` and `args`. See [`concat`](@ref) for a description of
`kwargs`.
"""
function concat!(data1::InferenceData, data::InferenceData...; kwargs...)
    data = Iterators.filter(x -> !isempty(x), data)
    arviz.concat(data1, data...; inplace = true, kwargs...)
    return data1
end

concat!(; kwargs...) = InferenceData()
