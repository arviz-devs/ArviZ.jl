# Implementation of Dataset is adapted from that of DimensionalData.DimStack
# in https://github.com/rafaqz/DimensionalData.jl under the MIT License.
# Copyright (c) 2019 Rafael Schouten <rafaelschouten@gmail.com>

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

In most cases, use [`convert_to_dataset`](@ref) or [`convert_to_constant_dataset`](@ref) or
to create a `Dataset` instead of directly using a constructor.
"""
struct Dataset{L,D<:Tuple,R<:Tuple,LD<:NamedTuple,M,LM<:NamedTuple} <:
       DimensionalData.AbstractDimStack{L}
    data::L
    dims::D
    refdims::R
    layerdims::LD
    metadata::M
    layermetadata::LM
end

Dataset(das::DimensionalData.AbstractDimArray...; kw...) = Dataset(das; kw...)
function Dataset(das::Tuple{Vararg{<:DimensionalData.AbstractDimArray}}; kw...)
    return Dataset(NamedTuple{DimensionalData.uniquekeys(das)}(das); kw...)
end
function Dataset(
    das::NamedTuple{<:Any,<:Tuple{Vararg{<:DimensionalData.AbstractDimArray}}};
    data=map(parent, das),
    dims=DimensionalData.combinedims(das...),
    layerdims=map(DimensionalData.basedims, das),
    refdims=(),
    metadata=DimensionalData.NoMetadata(),
    layermetadata=map(DimensionalData.metadata, das),
)
    return Dataset(data, dims, refdims, layerdims, metadata, layermetadata)
end
# Same sized arrays
function Dataset(
    data::NamedTuple,
    dims::Tuple;
    refdims=(),
    metadata=DimensionalData.NoMetadata(),
    layermetadata=map(_ -> DimensionalData.NoMetadata(), data),
)
    all(map(d -> axes(d) == axes(first(data)), data)) || throw(
        ArgumentError(
            "Arrays must have identical axes. For mixed dimensions, use DimArrays`"
        ),
    )
    layerdims = map(_ -> DimensionalData.basedims(dims), data)
    return Dataset(
        data,
        DimensionalData.format(dims, first(data)),
        refdims,
        layerdims,
        metadata,
        layermetadata,
    )
end
Dataset(data::Dataset) = data

Base.parent(data::Dataset) = getfield(data, :data)

PyObject(data::Dataset) = _to_xarray(data)

Base.convert(::Type{Dataset}, obj::PyObject) = _dataset_from_xarray(obj)
Base.convert(::Type{Dataset}, obj::Dataset) = obj
Base.convert(::Type{Dataset}, obj) = convert_to_dataset(obj)
function Base.convert(::Type{DimensionalData.DimStack}, obj::Dataset)
    return DimensionalData.DimStack(
        obj.data, obj.dims, obj.refdims, obj.layerdims, obj.metadata, obj.layermetadata
    )
end

@deprecate Base.getindex(data::Dataset, k::String) getindex(data, Symbol(k))

# Warning: this is not an API function and probably should be implemented abstractly upstream
DimensionalData.show_after(io, mime, ::Dataset) = nothing

function Base.show(io::IO, ::MIME"text/html", data::Dataset)
    obj = PyObject(data)
    (:_repr_html_ in propertynames(obj)) || return show(io, data)
    out = obj._repr_html_()
    out = replace(out, r"(<|&lt;)?xarray.Dataset(>|&gt;)?" => "Dataset (xarray.Dataset)")
    print(io, out)
    return nothing
end

attributes(data::Dataset) = getproperty(PyObject(data), :_attrs)

function setattribute!(data::Dataset, key, value)
    attrs = merge(attributes(data), Dict(key => value))
    setproperty!(PyObject(data), :_attrs, attrs)
    return attrs
end

@doc doc"""
    convert_to_dataset(obj; group = :posterior, kwargs...) -> Dataset

Convert a supported object to a `Dataset`.

In most cases, this function calls [`convert_to_inference_data`](@ref) and returns the
corresponding `group`.
"""
convert_to_dataset

function convert_to_dataset(obj; group=:posterior, kwargs...)
    group = Symbol(group)
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
    return Dataset(; data_vars=data, coords, attrs)
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
    return arviz.dict_to_dataset(data; attrs, kwargs...)
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
    dims = NamedTuple(Symbol(d) => collect(coords[d].values) for d in o.dims)
    return DimensionalData.DimArray(data, dims; name)
end

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
