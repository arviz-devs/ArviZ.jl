"""
    Dataset{L} <: DimensionalData.AbstractDimStack{L}

    Dataset(data::DimensionalData.AbstractDimArray...)
    Dataset(data::Tuple{Vararg{<:DimensionalData.AbstractDimArray}})
    Dataset(data::NamedTuple{Keys,Vararg{<:DimensionalData.AbstractDimArray}})
    Dataset(
        data::NamedTuple,
        dims::Tuple{Vararg{DimensionalData.Dimension}};
        metadata=DimensionalData.NoMetadata(),
    )

Collection of dimensional arrays with shared dimensions.

This type is an
[`DimensionalData.AbstractDimStack`](https://rafaqz.github.io/DimensionalData.jl/stable/api/#DimensionalData.AbstractDimStack)
that implements the same interface as `DimensionalData.DimStack` and has identical usage.

When a `Dataset` is passed to Python, it is converted to an `xarray.Dataset` without copying
the data. That is, the Python object shares the same memory as the Julia object. However,
if an `xarray.Dataset` is passed to Julia, its data must be copied.

In most cases, use [`convert_to_dataset`](@ref) or [`convert_to_constant_dataset`](@ref) or
to create a `Dataset` instead of directly using a constructor.
"""
struct Dataset{L,D<:DimensionalData.AbstractDimStack{L}} <:
       DimensionalData.AbstractDimStack{L}
    data::D
end

Dataset(args...; kwargs...) = Dataset(DimensionalData.DimStack(args...; kwargs...))
Dataset(data::Dataset) = data

Base.parent(data::Dataset) = getfield(data, :data)

PyObject(data::Dataset) = _to_xarray(data)

Base.convert(::Type{Dataset}, obj::PyObject) = _dataset_from_xarray(obj)
Base.convert(::Type{Dataset}, obj::Dataset) = obj
Base.convert(::Type{Dataset}, obj) = convert_to_dataset(obj)
function Base.convert(::Type{DimensionalData.DimStack}, data::Dataset)
    return convert(DimensionalData.DimStack, parent(data))
end

Base.propertynames(data::Dataset) = propertynames(parent(data))

Base.getproperty(data::Dataset, k::Symbol) = getproperty(parent(data), k)

@deprecate getindex(data::Dataset, k::String) getindex(data, Symbol(k))

# Warning: this is not an API function and probably should be implemented abstractly upstream
DimensionalData.show_after(io, mime, ::Dataset) = nothing

attributes(data::Dataset) = DimensionalData.metadata(data)

function setattribute!(data::Dataset, key::Symbol, value)
    setindex!(metadata(data), value, key)
    return value
end
@deprecate setattribute!(data::Dataset, key::AbstractString, value) setattribute!(
    data, Symbol(k), value
) false

@doc doc"""
    convert_to_dataset(obj; group = :posterior, kwargs...) -> Dataset

Convert a supported object to a `Dataset`.

In most cases, this function calls [`convert_to_inference_data`](@ref) and returns the
corresponding `group`.
"""
convert_to_dataset

function convert_to_dataset(obj; group::Symbol=:posterior, kwargs...)
    idata = convert_to_inference_data(obj; group, kwargs...)
    dataset = getproperty(idata, group)
    return dataset
end
convert_to_dataset(data::Dataset; kwargs...) = data

@doc doc"""
    convert_to_constant_dataset(obj::Dict; kwargs...) -> Dataset
    convert_to_constant_dataset(obj::NamedTuple; kwargs...) -> Dataset

Convert `obj` into a `Dataset`.

Unlike [`convert_to_dataset`](@ref), this is intended for containing constant parameters
such as observed data and constant data, and the first two dimensions are not required to be
the number of chains and draws.

# Keywords

- `coords::Dict{String,Vector}`: Map from named dimension to index names
- `dims::Dict{String,Vector{String}}`: Map from variable name to names of its dimensions
- `library::Any`: A library associated with the data to add to `attrs`.
- `attrs::Dict{String,Any}`: Global attributes to save on this dataset.
"""
convert_to_constant_dataset

function convert_to_constant_dataset(
    obj; coords=Dict(), dims=Dict(), library=nothing, attrs=Dict()
)
    base = arviz.data.base

    obj = _asstringkeydict(obj)
    coords = _asstringkeydict(coords)
    dims = _asstringkeydict(dims)
    attrs = _asstringkeydict(attrs)

    data = Dict{String,PyObject}()
    for (key, vals) in obj
        vals = _asarray(vals)
        val_dims = get(dims, key, nothing)
        (val_dims, val_coords) = base.generate_dims_coords(
            size(vals), key; dims=val_dims, coords
        )
        data[key] = xarray.DataArray(vals; dims=val_dims, coords=val_coords)
    end

    default_attrs = base.make_attrs()
    if library !== nothing
        default_attrs = merge(default_attrs, Dict("inference_library" => string(library)))
    end
    attrs = merge(default_attrs, attrs)
    return convert(Dataset, xarray.Dataset(; data_vars=data, coords, attrs))
end

@doc doc"""
    dict_to_dataset(data::Dict{String,Array}; kwargs...) -> Dataset

Convert a dictionary with data and keys as variable names to a [`Dataset`](@ref).

# Keywords

- `attrs::Dict{String,Any}`: Json serializable metadata to attach to the dataset, in
    addition to defaults.
- `library::String`: Name of library used for performing inference. Will be attached to the
    `attrs` metadata.
- `coords::Dict{String,Array}`: Coordinates for the dataset
- `dims::Dict{String,Vector{String}}`: Dimensions of each variable. The keys are variable
    names, values are vectors of coordinates.

# Examples

```@example
using ArviZ
ArviZ.dict_to_dataset(Dict("x" => randn(4, 100), "y" => randn(4, 100)))
```
"""
dict_to_dataset

function dict_to_dataset(data; library=nothing, attrs=Dict(), kwargs...)
    if library !== nothing
        attrs = merge(attrs, Dict("inference_library" => string(library)))
    end
    return convert(Dataset, arviz.dict_to_dataset(data; attrs, kwargs...))
end

@doc doc"""
    dataset_to_dict(ds::Dataset) -> Tuple{Dict{String,Array},NamedTuple}

Convert a `Dataset` to a dictionary of `Array`s. The function also returns keyword arguments
to [`dict_to_dataset`](@ref).
"""
dataset_to_dict

function dataset_to_dict(ds::Dataset)
    data = Dict(pairs(DimensionalData.data(ds)))
    attrs = Dict(pairs(DimensionalData.metadata(ds)))
    dims = Dict(pairs(map(collect ∘ DimensionalData.name, DimensionalData.layerdims(ds))))
    coords = Dict(Symbol(name(d)) => collect(d) for d in DimensionalData.dims(ds))
    return data, (; attrs, coords, dims)
end

function _dataset_from_xarray(o::PyObject)
    pyisinstance(o, xarray.Dataset) ||
        throw(ArgumentError("argument is not an `xarray.Dataset`."))
    var_names = collect(o.data_vars)
    data = [_dimarray_from_xarray(getindex(o, name)) for name in var_names]
    metadata = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in o.attrs)
    return Dataset(data...; metadata)
end

function _dimarray_from_xarray(o::PyObject)
    pyisinstance(o, xarray.DataArray) ||
        throw(ArgumentError("argument is not an `xarray.DataArray`."))
    name = Symbol(o.name)
    data = o.to_numpy()
    coords = ArviZ.PyCall.PyDict(o.coords)
    dims = Tuple(map(d -> _wrap_dims(Symbol(d), _process_dims(coords[d].values)), o.dims))
    return DimensionalData.DimArray(data, dims; name)
end

_process_dims(dims) = collect(map(identity, dims))
# NOTE: sometimes strings fail to convert to Julia types, so we try to force them here
function _process_dims(dims::AbstractVector{<:PyObject})
    return collect(map(Base.Fix1(convert, String), dims))
end

# wrap dims in a `Dim`, converting to an AbstractRange if possible
function _wrap_dims(name::Symbol, dims::AbstractVector{<:Real})
    D = DimensionalData.Dim{name}
    start = dims[begin]
    stop = dims[end]
    n = length(dims)
    step = (stop - start) / (n - 1)
    isrange = all(Iterators.drop(eachindex(dims), 1)) do i
        return (dims[i] - dims[i - 1]) ≈ step
    end
    return if isrange
        if step == 1
            D(UnitRange(start, stop))
        else
            D(range(start, stop, n))
        end
    else
        D(dims)
    end
end
_wrap_dims(name::Symbol, dims::AbstractVector) = DimensionalData.Dim{name}(dims)

function _to_xarray(data::DimensionalData.AbstractDimStack)
    data_vars = Dict(pairs(map(_to_xarray, DimensionalData.layers(data))))
    attrs = Dict(pairs(DimensionalData.metadata(data)))
    return PyCall.pycall(xarray.Dataset, PyObject, data_vars; attrs)
end

function _to_xarray(data::DimensionalData.AbstractDimArray)
    var_name = DimensionalData.name(data)
    data_dims = DimensionalData.dims(data)
    dims = collect(DimensionalData.name(data_dims))
    coords = Dict(zip(dims, collect.(data_dims)))
    default_dims = String[]
    values = parent(data)
    return arviz.numpy_to_data_array(values; var_name, dims, coords, default_dims)
end
