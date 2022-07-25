@forwardfun extract_dataset
convert_result(::typeof(extract_dataset), result, args...) = convert(Dataset, result)

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
