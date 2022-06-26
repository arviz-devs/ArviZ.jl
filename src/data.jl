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
struct InferenceData{group_names,group_types<:Tuple{Vararg{Dataset}}}
    groups::NamedTuple{group_names,group_types}
    function InferenceData(groups::NamedTuple{group_names,<:Tuple{Vararg{Dataset}}}) where {group_names}
        group_names_ordered = _reorder_group_names(Val{group_names}())
        groups_ordered = NamedTuple{group_names_ordered}(groups)
        return new{group_names_ordered,typeof(values(groups_ordered))}(groups_ordered)
    end
end
InferenceData(; kwargs...) = InferenceData(NamedTuple(kwargs))
InferenceData(data::InferenceData) = data

Base.parent(data::InferenceData) = getfield(data, :groups)

Base.convert(::Type{InferenceData}, obj::InferenceData) = obj
Base.convert(::Type{InferenceData}, obj) = convert_to_inference_data(obj)

Base.convert(::Type{NamedTuple}, data::InferenceData) = NamedTuple(data)
NamedTuple(data::InferenceData) = parent(data)

# these 3 interfaces ensure InferenceData behaves like a NamedTuple

# properties interface
Base.propertynames(data::InferenceData) = propertynames(parent(data))
Base.getproperty(data::InferenceData, k::Symbol) = getproperty(parent(data), k)

# indexing interface
Base.getindex(data::InferenceData, k) = parent(data)[k]
function Base.setindex(data::InferenceData, v, k::Symbol)
    return InferenceData(Base.setindex(parent(data), v, k))
end

# iteration interface
Base.keys(data::InferenceData) = keys(parent(data))
Base.haskey(data::InferenceData, k::Symbol) = haskey(parent(data), k)
Base.values(data::InferenceData) = values(parent(data))
Base.pairs(data::InferenceData) = pairs(parent(data))
Base.length(data::InferenceData) = length(parent(data))
Base.iterate(data::InferenceData, i...) = iterate(parent(data), i...)
Base.eltype(data::InferenceData) = eltype(parent(data))

function Base.map(f, data::InferenceData)
    ret = map(f, parent(data))
    # if output can be an InferenceData, then make it so
    ret isa NamedTuple{<:Any,<:Tuple{Vararg{Dataset}}} && return InferenceData(ret)
    return ret
end

@forwardfun extract_dataset
convert_result(::typeof(extract_dataset), result, args...) = convert(Dataset, result)

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
    groups(data::InferenceData)

Get the groups in `data` as a named tuple mapping symbols to [`Datasets`](@ref).
"""
groups(data::InferenceData) = parent(data)

"""
    groupnames(data::InferenceData)

Get the names of the groups (datasets) in `data` as a tuple of symbols.
"""
groupnames(data::InferenceData) = keys(groups(data))

"""
    hasgroup(data::InferenceData, name::Symbol) -> Bool

Return `true` if a group with name `name` is stored in `data`.
"""
hasgroup(data::InferenceData, name::Symbol) = haskey(data, name)

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
    dicts_all = (; posterior, dicts...)
    dicts = NamedTuple(filter(x -> !(x[2] === nothing || isempty(x[2])), pairs(dicts_all)))
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

@generated function _reorder_group_names(::Val{names}) where {names}
    return Tuple(sort(collect(names); by = k -> SUPPORTED_GROUPS_DICT[k]))
end

@generated _keys_and_types(::NamedTuple{keys,types}) where {keys,types} = (keys, types)

# python interop

function PyObject(data::InferenceData)
    return pycall(arviz.InferenceData, PyObject; map(PyObject, groups(data))...)
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
