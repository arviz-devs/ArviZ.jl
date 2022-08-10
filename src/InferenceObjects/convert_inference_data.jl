"""
    convert(::Type{InferenceData}, obj)

Convert `obj` to an `InferenceData`.

`obj` can be any type for which [`convert_to_inference_data`](@ref) is defined.
"""
Base.convert(::Type{InferenceData}, obj) = convert_to_inference_data(obj)
Base.convert(::Type{InferenceData}, obj::InferenceData) = obj
Base.convert(::Type{NamedTuple}, data::InferenceData) = NamedTuple(data)
NamedTuple(data::InferenceData) = parent(data)

"""
    convert_to_inference_data(obj; group, kwargs...) -> InferenceData

Convert a supported object to an [`InferenceData`](@ref) object.

If `obj` converts to a single dataset, `group` specifies which dataset in the resulting
`InferenceData` that is.

See [`convert_to_dataset`](@ref)

# Arguments

  - `obj` can be many objects. Basic supported types are:

      + [`InferenceData`](@ref): return unchanged
      + [`Dataset`](@ref)/`DimensionalData.AbstractDimStack`: add to `InferenceData` as the only
        group
      + `NamedTuple`/`AbstractDict`: create a `Dataset` as the only group
      + `AbstractArray{<:Real}`: create a `Dataset` as the only group, given an arbitrary
        name, if the name is not set

More specific types may be documented separately.

# Keywords

  - `group::Symbol = :posterior`: If `obj` converts to a single dataset, assign the resulting
    dataset to this group.
  - `dims`: a collection mapping variable names to collections of objects containing
    dimension names. Acceptable such objects are:
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
