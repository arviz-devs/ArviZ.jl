"""
    InferenceData

Loose wrapper around `arviz.InferenceData`, which is a container for inference
data storage using xarray.

# Constructor

    InferenceData(o::PyObject)

wraps an `arviz.InferenceData`. To create an `InferenceData`, use the exported
`from_xyz` functions or `convert_to_inference_data`.
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
    out = replace(out, r"Inference data" => "InferenceData")
    print(io, out)
end

@forwardfun convert_to_inference_data

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

    idata = InferenceData()
    for (name, dict) in pairs(dicts)
        (isnothing(dict) || isempty(dict)) && continue
        dataset = dict_to_dataset(dict; attrs = attrs, coords = coords, dims = dims)
        concat!(idata, InferenceData(; (name => dataset,)...))
    end

    return idata
end

function concat(args...; kwargs...)
    ret = arviz.concat(args...; kwargs...)
    ret === nothing && return args[1]
    return ret
end

Base.Docs.getdoc(::typeof(concat)) = Base.Docs.getdoc(arviz.concat)

"""
    concat!(data::InferenceData, args::InferenceData...; kwargs...)

In-place version of `concat`, where `data` is modified to contain the
concatenation of `data` and `args`. See `concat` for a description of
`kwargs`.
"""
function concat!(data, args...; kwargs...)
    kwargs = merge((; kwargs...), (; inplace = true))
    concat(data, args...; kwargs...)
    return data
end
