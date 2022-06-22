const SUPPORTED_GROUPS = Symbol[]
const SUPPORTED_GROUPS_DICT = Dict{Symbol,Int}()

"""
    InferenceData(; kwargs...)

Container for inference data storage using DimensionalData.

`InferenceData` can be constructed either from an `arviz.InferenceData` or from multiple
[`Dataset`](@ref)s assigned to groups specified as `kwargs`.

Instead of directly creating an `InferenceData`, use the exported `from_xyz` functions or
[`convert_to_inference_data`](@ref).
"""
struct InferenceData
    groups::Dict{Symbol,Dataset}
end
InferenceData(; kwargs...) = InferenceData(Dict(kwargs))
InferenceData(data::InferenceData) = data

function PyObject(data::InferenceData)
    return pycall(arviz.InferenceData, PyObject; groups(data)...)
end

function Base.convert(::Type{InferenceData}, obj::PyObject)
    pyisinstance(obj, arviz.InferenceData) ||
        throw(ArgumentError("argument is not an `arviz.InferenceData`."))
    group_names = obj.groups()
    groups = Dict(Symbol(name) => getindex(obj, name) for name in group_names)
    return InferenceData(groups)
end
Base.convert(::Type{InferenceData}, obj) = convert_to_inference_data(obj)
Base.convert(::Type{InferenceData}, obj::InferenceData) = obj

Base.hash(data::InferenceData) = hash(groups(data))

Base.propertynames(data::InferenceData) = sort!(collect(keys(groups(data))))

Base.hasproperty(data::InferenceData, k::Symbol) = hasgroup(data, k)

Base.getproperty(data::InferenceData, k::Symbol) = getindex(groups(data), k)

function Base.setproperty!(data::InferenceData, k::Symbol, ds::Dataset)
    groups(data)[k] = ds
    return ds
end

Base.delete!(data::InferenceData, name::Symbol) = delete!(groups(data), name)

@forwardfun extract_dataset

@deprecate (data1::InferenceData + data2::InferenceData) convert(
    InferenceData, PyObject(data1) + PyObject(data2)
)

function Base.show(io::IO, ::MIME"text/plain", data::InferenceData)
    print(io, "InferenceData with groups:")
    prefix = "\n    > "
    for name in groupnames(data)
        print(io, prefix, name)
    end
    return nothing
end
function Base.show(io::IO, ::MIME"text/html", data::InferenceData)
    obj = PyObject(data)
    (:_repr_html_ in propertynames(obj)) || return show(io, data)
    out = obj._repr_html_()
    out = replace(out, r"arviz.InferenceData" => "InferenceData")
    out = replace(out, r"(<|&lt;)?xarray.Dataset(>|&gt;)?" => "Dataset")
    print(io, out)
    return nothing
end

"""
    groupnames(data::InferenceData) -> Vector{Symbol}

Get the names of the groups (datasets) in `data`.
"""
function groupnames(data::InferenceData)
    return sort!(collect(keys(groups(data))); by=k -> SUPPORTED_GROUPS_DICT[k])
end

"""
    groups(data::InferenceData) -> Dict{Symbol,Dataset}

Get the groups in `data` as a dictionary mapping names to datasets.
"""
groups(data::InferenceData) = getfield(data, :groups)

"""
    hasgroup(data::InferenceData, name::Symbol) -> Bool

Return `true` if a group with name `name` is stored in `data`.
"""
hasgroup(data::InferenceData, name::Symbol) = haskey(groups(data), name)

Base.isempty(data::InferenceData) = isempty(groups(data))

@forwardfun convert_to_inference_data

convert_to_inference_data(::Nothing; kwargs...) = InferenceData()

function convert_to_dataset(data::InferenceData; group=:posterior, kwargs...)
    return getproperty(data, Symbol(group))
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
function _from_dict(posterior=nothing; attrs=Dict(), coords=nothing, dims=nothing, dicts...)
    dicts = (posterior=posterior, dicts...)

    groups = Dict{Symbol,Dataset}()
    for (name, dict) in pairs(dicts)
        (dict === nothing || isempty(dict)) && continue
        dataset = dict_to_dataset(dict; attrs, coords, dims)
        groups[name] = dataset
    end

    idata = InferenceData(groups)
    return idata
end

@doc forwarddoc(:concat) concat

function concat(data::InferenceData...; kwargs...)
    return arviz.concat(data...; inplace=false, kwargs...)
end

Docs.getdoc(::typeof(concat)) = forwardgetdoc(:concat)

@doc doc"""
    concat!(data1::InferenceData, data::InferenceData...; kwargs...) -> InferenceData

In-place version of `concat`, where `data1` is modified to contain the concatenation of
`data` and `args`. See [`concat`](@ref) for a description of `kwargs`.
"""
concat!

function concat!(data::InferenceData, other_data::InferenceData...; kwargs...)
    arviz.concat(data, other_data...; inplace=true, kwargs...)
    return data
end
concat!(; kwargs...) = InferenceData()

function rekey(data::InferenceData, keymap)
    keymap = Dict([Symbol(k) => Symbol(v) for (k, v) in keymap])
    dnames = groupnames(data)
    groups_new = Dict{Symbol,Dataset}()
    for k in dnames
        knew = get(keymap, k, k)
        groups_new[knew] = getproperty(data, k)
    end
    return InferenceData(groups_new)
end
