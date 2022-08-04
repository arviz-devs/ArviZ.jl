"""
    Dataset{L} <: DimensionalData.AbstractDimStack{L}

Container of dimensional arrays sharing some dimensions.

This type is an
[`DimensionalData.AbstractDimStack`](https://rafaqz.github.io/DimensionalData.jl/stable/api/#DimensionalData.AbstractDimStack)
that implements the same interface as `DimensionalData.DimStack` and has identical usage.

When a `Dataset` is passed to Python, it is converted to an `xarray.Dataset` without copying
the data. That is, the Python object shares the same memory as the Julia object. However,
if an `xarray.Dataset` is passed to Julia, its data must be copied.

# Constructors

    Dataset(data::DimensionalData.AbstractDimArray...)
    Dataset(data::Tuple{Vararg{<:DimensionalData.AbstractDimArray}})
    Dataset(data::NamedTuple{Keys,Vararg{<:DimensionalData.AbstractDimArray}})
    Dataset(
        data::NamedTuple,
        dims::Tuple{Vararg{DimensionalData.Dimension}};
        metadata=DimensionalData.NoMetadata(),
    )

In most cases, use [`convert_to_dataset`](@ref) to create a `Dataset` instead of directly
using a constructor.
"""
struct Dataset{L,D<:DimensionalData.AbstractDimStack{L}} <:
       DimensionalData.AbstractDimStack{L}
    data::D
end

Dataset(args...; kwargs...) = Dataset(DimensionalData.DimStack(args...; kwargs...))
Dataset(data::Dataset) = data

Base.parent(data::Dataset) = getfield(data, :data)

Base.propertynames(data::Dataset) = keys(data)

Base.getproperty(data::Dataset, k::Symbol) = getindex(data, k)

function setattribute!(data::Dataset, k::Symbol, value)
    setindex!(DimensionalData.metadata(data), value, k)
    return value
end
@deprecate setattribute!(data::Dataset, k::AbstractString, value) setattribute!(
    data, Symbol(k), value
) false

Base.convert(::Type{Dataset}, obj::Dataset) = obj
Base.convert(::Type{Dataset}, obj) = convert_to_dataset(obj)

@doc doc"""
    convert_to_dataset(obj; group = :posterior, kwargs...) -> Dataset

Convert a supported object to a `Dataset`.

In most cases, this function calls [`convert_to_inference_data`](@ref) and returns the
corresponding `group`.
"""
function convert_to_dataset end

function convert_to_dataset(obj; group::Symbol=:posterior, kwargs...)
    idata = convert_to_inference_data(obj; group, kwargs...)
    dataset = getproperty(idata, group)
    return dataset
end
convert_to_dataset(data::Dataset; kwargs...) = data

"""
    namedtuple_to_dataset(data; kwargs...) -> Dataset

Convert `NamedTuple` mapping variable names to arrays to a [`Dataset`](@ref).

# Keywords

  - `attrs`: a Symbol-indexable collection of metadata to attach to the dataset, in addition
    to defaults. Values should be JSON serializable.

  - `library::Union{String,Module}`: library used for performing inference. Will be attached
    to the `attrs` metadata.
  - `dims`: a collection mapping variable names to collections of objects containing dimension
    names. Acceptable such objects are:
    
      + `Symbol`: dimension name
      + `Type{<:DimensionsionalData.Dimension}`: dimension type
      + `DimensionsionalData.Dimension`: dimension, potentially with indices
      + `Nothing`: no dimension name provided, dimension name is automatically generated
  - `coords`: a collection indexable by dimension name specifying the indices of the given
    dimension. If indices for a dimension in `dims` are provided, they are used even if
    the dimension contains its own indices. If a dimension is missing, its indices are
    automatically generated.
"""
function namedtuple_to_dataset end
function namedtuple_to_dataset(
    data; attrs=(;), library=nothing, dims=(;), coords=(;), default_dims=DEFAULT_SAMPLE_DIMS
)
    dim_arrays = map(keys(data)) do var_name
        var_data = data[var_name]
        var_dims = get(dims, var_name, ())
        return array_to_dimarray(var_data, var_name; dims=var_dims, coords, default_dims)
    end
    attributes = merge(default_attributes(library), attrs)
    metadata = OrderedDict{Symbol,Any}(pairs(attributes))
    return Dataset(dim_arrays...; metadata)
end

"""
    array_to_dimarray(array, name; kwargs...) -> DimensionalData.AbstractDimArray

Convert `array` to a `AbstractDimArray` with name `name`.

If `array` is already an `AbstractDimArray`, then it is returned without modification.
See [`generate_dims`](@ref) for a description of `kwargs`.
"""
function array_to_dimarray end
function array_to_dimarray(array::DimensionalData.AbstractDimArray, name; kwargs...)
    return DimensionalData.rebuild(array; name)
end
function array_to_dimarray(data, name; dims=(), coords=(;), default_dims=())
    array = if ndims(data) < 2 && has_all_sample_dims(default_dims)
        reshape(data, 1, :)
    else
        data
    end
    array_dims = generate_dims(array, name; dims, coords, default_dims)
    return DimensionalData.DimArray(array, array_dims; name)
end

has_all_sample_dims(dims) = all(Dimensions.hasdim(dims, DEFAULT_SAMPLE_DIMS))

"""
    default_attributes(library=nothing) -> NamedTuple

Generate default attributes metadata for a dataset generated by inference library `library`.

`library` may be a `String` or a `Module`.
"""
function default_attributes(library=nothing)
    return (
        created_at=current_time_iso(),
        arviz_version=string(package_version(ArviZ)),
        arviz_language="julia",
        library_attributes(library)...,
    )
end

library_attributes(library) = (; inference_library=string(library))
library_attributes(::Nothing) = (;)
function library_attributes(library::Module)
    return (
        inference_library=string(library),
        inference_library_version=string(package_version(library)),
    )
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
function generate_dims end
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

# DimensionalData interop

for f in [:data, :dims, :refdims, :metadata, :layerdims, :layermetadata]
    @eval begin
        DimensionalData.$(f)(ds::Dataset) = DimensionalData.$(f)(parent(ds))
    end
end

# Warning: this is not an API function and probably should be implemented abstractly upstream
DimensionalData.show_after(io, mime, ::Dataset) = nothing

attributes(data::DimensionalData.AbstractDimStack) = DimensionalData.metadata(data)

Base.convert(T::Type{<:DimensionalData.DimStack}, data::Dataset) = convert(T, parent(data))

function DimensionalData.rebuild(data::Dataset; kwargs...)
    return Dataset(DimensionalData.rebuild(parent(data); kwargs...))
end

# python interop

PyObject(data::Dataset) = _to_xarray(data)

Base.convert(::Type{Dataset}, obj::PyObject) = Dataset(_dimstack_from_xarray(obj))
