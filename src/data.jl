const SUPPORTED_GROUPS = Symbol[]
const SUPPORTED_GROUPS_DICT = Dict{Symbol,Int}()

"""
    InferenceData(groups)
    InferenceData(; groups...)

Container for inference data storage using DimensionalData.

`InferenceData` can be constructed from either a `NamedTuple` or pairs mapping a group name
to a corresponding [`Dataset`](@ref).

Instead of directly creating an `InferenceData`, use the exported `from_xyz` functions or
[`convert_to_inference_data`](@ref).
"""
struct InferenceData{group_names,group_types<:Tuple{Vararg{Dataset}}}
    groups::NamedTuple{group_names,group_types}
    function InferenceData(
        groups::NamedTuple{group_names,<:Tuple{Vararg{Dataset}}}
    ) where {group_names}
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

@forwardfun extract_dataset
convert_result(::typeof(extract_dataset), result, args...) = convert(Dataset, result)

function Base.show(io::IO, ::MIME"text/plain", data::InferenceData)
    print(io, "InferenceData with groups:")
    prefix = "\n  > "
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

Get the groups in `data` as a named tuple mapping symbols to [`Dataset`](@ref)s.
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

"""
    convert_to_inference_data(obj; group, kwargs...) -> InferenceData

Convert a supported object to an [`InferenceData`](@ref) object.

If `obj` converts to a single dataset, `group` specifies which dataset in the resulting
`InferenceData` that is.

# Arguments

  - `obj` can be many objects. Basic supported types are:
    
      + [`InferenceData`](@ref): return unchanged
      + `AbstractString`: attempt to load a NetCDF file from disk
      + [`Dataset`](@ref)/`DimensionalData.AbstractDimStack`: add to `InferenceData` as the only
        group
      + `NamedTuple`/`AbstractDict`: create a `Dataset` as the only group
      + `AbstractArray{<:Real}`: create a `Dataset` as the only group, given an arbitrary
        name, if the name is not set
      + `PyCall.PyObject`: forward object to Python ArviZ for conversion

More specific types are documented separately.

# Keywords

  - `group::Symbol = :posterior`: If `obj` converts to a single dataset, assign the resulting
    dataset to this group.

  - `dims`: a collection mapping variable names to collections of objects containing dimension names. Acceptable such objects are:
    
      + `Symbol`: dimension name
      + `Type{<:DimensionsionalData.Dimension}`: dimension type
      + `DimensionsionalData.Dimension`: dimension, potentially with indices
      + `Nothing`: no dimension name provided, dimension name is automatically generated
  - `coords`: a collection indexable by dimension name specifying the indices of the given
    dimension. If indices for a dimension in `dims` are provided, they are used even if
    the dimension contains its own indices. If a dimension is missing, its indices are
    automatically generated.
  - `kwargs`: remaining keywords forwarded to converter functions
"""
function convert_to_inference_data end

convert_to_inference_data(data::InferenceData; kwargs...) = data
function convert_to_inference_data(stack::DimensionalData.AbstractDimStack; kwargs...)
    return convert_to_inference_data(Dataset(stack); kwargs...)
end
function convert_to_inference_data(data::Dataset; group=:posterior, kwargs...)
    return convert_to_inference_data(InferenceData(; group => data); kwargs...)
end
function convert_to_inference_data(data::AbstractDict{Symbol}; kwargs...)
    return convert_to_inference_data(NamedTuple(data); kwargs...)
end
function convert_to_inference_data(var_data::AbstractArray{<:Real}; kwargs...)
    data = (; default_var_name(var_data) => var_data)
    return convert_to_inference_data(data; kwargs...)
end
function convert_to_inference_data(filename::AbstractString; kwargs...)
    return from_netcdf(filename)
end
function convert_to_inference_data(
    data::NamedTuple{<:Any,<:Tuple{Vararg{AbstractArray{<:Real}}}};
    group=:posterior,
    kwargs...,
)
    ds = namedtuple_to_dataset(data; kwargs...)
    return convert_to_inference_data(ds; group)
end

"""
    default_var_name(data) -> Symbol

Return the default name for the variable whose values are stored in `data`.
"""
default_var_name(data) = :x
function default_var_name(data::DimensionalData.AbstractDimArray)
    name = DimensionalData.name(data)
    name isa Symbol && return name
    name isa AbstractString && !isempty(name) && return Symbol(name)
    return default_var_name(parent(data))
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
    return Tuple(sort(collect(names); by=k -> SUPPORTED_GROUPS_DICT[k]))
end

@generated _keys_and_types(::NamedTuple{keys,types}) where {keys,types} = (keys, types)

# python interop

function PyObject(data::InferenceData)
    return pycall(arviz.InferenceData, PyObject; map(PyObject, groups(data))...)
end

function convert_to_inference_data(obj::PyObject; dims=nothing, coords=nothing, kwargs...)
    if pyisinstance(obj, arviz.InferenceData)
        group_names = obj.groups()
        groups = (
            Symbol(name) => convert(Dataset, getindex(obj, name)) for name in group_names
        )
        return InferenceData(; groups...)
    else
        # Python ArviZ requires dims and coords be dicts matching to vectors
        pydims = dims === nothing ? dims : Dict(k -> collect(dims[k]) for k in keys(dims))
        pycoords =
            dims === nothing ? dims : Dict(k -> collect(coords[k]) for k in keys(coords))
        return arviz.convert_to_inference_data(obj; dims=pydims, coords=pycoords, kwargs...)
    end
end
