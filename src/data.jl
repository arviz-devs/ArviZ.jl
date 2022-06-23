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
struct InferenceData{group_names,groups}
    groups::NamedTuple{group_names,groups}
end
InferenceData(; kwargs...) = InferenceData(NamedTuple(kwargs))
InferenceData(data::InferenceData) = data
function InferenceData(data::NamedTuple)
    groups_reordered = _reorder_groups(data)
    group_names, groups = _keys_and_types(groups_reordered)
    return InferenceData{group_names,groups}(groups_reordered)
end

function PyObject(data::InferenceData)
    return pycall(arviz.InferenceData, PyObject; map(_to_xarray, groups(data))...)
end

function Base.convert(::Type{InferenceData}, obj::PyObject)
    pyisinstance(obj, arviz.InferenceData) ||
        throw(ArgumentError("argument is not an `arviz.InferenceData`."))
    group_names = obj.groups()
    groups = NamedTuple(
        Symbol(name) => convert(Dataset, getindex(obj, name)) for name in group_names
    )
    return InferenceData(groups)
end
Base.convert(::Type{InferenceData}, obj) = convert_to_inference_data(obj)
Base.convert(::Type{InferenceData}, obj::InferenceData) = obj

Base.hash(data::InferenceData) = hash(groups(data))

Base.propertynames(data::InferenceData) = groupnames(data)

Base.hasproperty(data::InferenceData, k::Symbol) = hasgroup(data, k)

Base.getproperty(data::InferenceData, k::Symbol) = getproperty(groups(data), k)

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
function Base.show(io::IO, mime::MIME"text/html", data::InferenceData)
    show(io, mime, HTML("<div>InferenceData"))
    for (name, group) in pairs(groups(data))
        show(io, mime, HTML("""
        <details>
        <summary>$name</summary>
        <pre><code>$(sprint(show, "text/plain", group))</code></pre>
        </details>
        """))
    end
    return show(io, mime, HTML("</div>"))
end

"""
    groupnames(data::InferenceData) -> Vector{Symbol}

Get the names of the groups (datasets) in `data`.
"""
groupnames(data::InferenceData) = keys(groups(data))

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
function convert_to_inference_data(data::Dataset; group::Symbol=:posterior, kwargs...)
    return InferenceData(; group=data)
end

function convert_to_dataset(data::InferenceData; group::Symbol=:posterior, kwargs...)
    return getproperty(data, group)
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
    dicts = filter(d -> !(d === nothing || isempty(d)), (; posterior, dicts...))
    groups = map(d -> dict_to_dataset(d; attrs, coords, dims), dicts)
    idata = InferenceData(groups)
    return idata
end

@doc forwarddoc(:concat) concat

function concat(data::InferenceData...; kwargs...)
    return arviz.concat(data...; inplace=false, kwargs...)
end

Docs.getdoc(::typeof(concat)) = forwardgetdoc(:concat)

function Base.merge(data::InferenceData, other_data::InferenceData...)
    return InferenceData(Base.merge(groups(data), map(groups, other_data)...))
end

function rekey(data::InferenceData, keymap)
    groups_old = groups(data)
    names_new = map(k -> get(keymap, k, k), propertynames(groups_old))
    groups_new = NamedTuple{names_new}(Tuple(groups_old))
    return InferenceData(groups_new)
end

_reorder_groups(group::NamedTuple) = _reorder_groups_type(typeof(group))(group)

@generated function _reorder_groups_type(::Type{<:NamedTuple{names}}) where {names}
    return NamedTuple{Tuple(sort(collect(names); by = k -> SUPPORTED_GROUPS_DICT[k]))}
end

@generated _keys_and_types(::NamedTuple{keys,types}) where {keys,types} = (keys, types)
