has_all_sample_dims(dims) = all(Dimensions.hasdim(dims, DEFAULT_SAMPLE_DIMS))

"""
    as_dimension(dim, coords, length) -> DimensionsionalData.Dimension

Convert `dim`, `coords`, and `length` to a `Dimension` object.

# Arguments

  - `dim`: An object specifying the name and potentially indices of a dimension. Can be the
    following types:
    
      + `Symbol`: dimension name.
      + `Type{<:DimensionsionalData.Dimension}`: dimension type
      + `DimensionsionalData.Dimension`: dimension, potentially with indices

  - `coords`: a collection indexable by dimension name specifying the indices of the given
    dimension. If indices are provided, they are used even if `dim` contains its own
    indices. If a dimension is missing, its indices are automatically generated.
  - `length`: the length of the dimension. If `coords` and `dim` indices are not provided,
    then the indices `1:n` are used.
"""
function as_dimension end
function as_dimension(dim::Dimensions.Dimension, coords, n)
    name = Dimensions.name(dim)
    haskey(coords, name) && return Dimensions.rebuild(dim, coords[name])
    Dimensions.val(dim) isa Colon && return Dimensions.rebuild(dim, 1:n)
    return dim
end
function as_dimension(dim::Type{<:Dimensions.Dimension}, coords, n)
    return as_dimension(dim(1:n), coords, n)
end
function as_dimension(dim::Symbol, coords, n)
    return as_dimension(Dimensions.rebuild(Dimensions.key2dim(dim), 1:n), coords, n)
end

"""
    generate_dims(array, name; dims, coords, default_dims)

Generate `DimensionsionalData.Dimension` objects for each dimension of `array`.

`name` indicates the name of the variable represented by array.

# Keywords

  - `dims`: A collection of objects indicating dimension names. If any dimensions are not
    provided, their names are automatically generated. Acceptable types of entries are:
    
      + `Symbol`: dimension name
      + `Type{<:DimensionsionalData.Dimension}`: dimension type
      + `DimensionsionalData.Dimension`: dimension, potentially with indices
      + `Nothing`: no dimension name provided, dimension name is automatically generated

  - `coords`: a collection indexable by dimension name specifying the indices of the given
    dimension. If indices for a dimension in `dims` are provided, they are used even if
    the dimension contains its own indices. If a dimension is missing, its indices are
    automatically generated.
  - `default_dims`: A collection of dims to be prepended to `dims` whose elements have the
    same constraints.
"""
function generate_dims(array, name; dims=(), coords=(;), default_dims=())
    num_default_dims = length(default_dims)
    length(dims) + num_default_dims > ndims(array) && @error "blah"
    dims_named = ntuple(ndims(array) - length(default_dims)) do i
        dim = get(dims, i, nothing)
        dim === nothing && return Symbol("$(name)_dim_$(i)")
        return dim
    end
    dims_all = (default_dims..., dims_named...)
    dims_with_coords = ntuple(ndims(array)) do i
        return as_dimension(dims_all[i], coords, size(array, i))
    end
    return Dimensions.format(dims_with_coords, array)
end

"""
    array_to_dimarray(array, name; kwargs...) -> DimensionalData.AbstractDimArray

Convert `array` to a `AbstractDimArray` with name `name`.

If `array` is already an `AbstractDimArray`, then it is returned without modification.
See [`generate_dims`](@ref) for a description of `kwargs`.
"""
function array_to_dimarray(data, name; dims=(), coords=(;), default_dims=())
    array = if ndims(data) < 2 && has_all_sample_dims(default_dims)
        reshape(data, 1, :)
    else
        data
    end
    array_dims = generate_dims(array, name; dims, coords, default_dims)
    return DimensionalData.DimArray(array, array_dims; name)
end
function array_to_dimarray(array::DimensionalData.AbstractDimArray, name; kwargs...)
    return DimensionalData.rebuild(array; name)
end

"""
    AsSlice{T<:Dimensions.Selector} <: Dimensions.Selector{T}

    AsSlice(selector)

Selector that ensures selected indices are arrays so that slicing occurs.

This is useful to ensure that selecting a single index still returns an array.
"""
struct AsSlice{T<:Dimensions.Selector} <: Dimensions.Selector{T}
    val::T
end

function Dimensions.selectindices(l::Dimensions.LookupArray, sel::AsSlice; kw...)
    i = Dimensions.selectindices(l, Dimensions.val(sel); kw...)
    inds = i isa AbstractVector ? i : [i]
    return inds
end

"""
    index_to_indices(index)

Convert `index` to a collection of indices or a selector representing such a collection.
"""
index_to_indices(i) = i
index_to_indices(i::Int) = [i]
index_to_indices(sel::Dimensions.Selector) = AsSlice(sel)
