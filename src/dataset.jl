"""
    Dataset(::PyObject)
    Dataset(; data_vars = nothing, coords = nothing, attrs = nothing)

Loose wrapper around `xarray.Dataset`, mostly used for dispatch.

# Keywords

- `data_vars::Dict{String,Any}`: Dict mapping variable names to
    + `Vector`: Data vector. Single dimension is named after variable.
    + `Tuple{String,Vector}`: Dimension name and data vector.
    + `Tuple{NTuple{N,String},Array{T,N}} where {N,T}`: Dimension names and data array.
- `coords::Dict{String,Any}`: Dict mapping dimension names to index names. Possible
    arguments has same form as `data_vars`.
- `attrs::Dict{String,Any}`: Global attributes to save on this dataset.

In most cases, use [`convert_to_dataset`](@ref) or [`convert_to_constant_dataset`](@ref) or
to create a `Dataset` instead of directly using a constructor.
"""
struct Dataset
    o::PyObject

    function Dataset(o::PyObject)
        pyisinstance(o, xarray.Dataset) && return new(o)
        throw(ArgumentError("$o is not an `xarray.Dataset`."))
    end
end

Dataset(; kwargs...) = xarray.Dataset(; kwargs...)
@inline Dataset(data::Dataset) = data

@inline PyObject(data::Dataset) = getfield(data, :o)

Base.convert(::Type{Dataset}, obj::PyObject) = Dataset(obj)
Base.convert(::Type{Dataset}, obj::Dataset) = obj
Base.convert(::Type{Dataset}, obj) = convert_to_dataset(obj)

Base.hash(data::Dataset) = hash(PyObject(data))

Base.propertynames(data::Dataset) = propertynames(PyObject(data))

function Base.getproperty(data::Dataset, name::Symbol)
    o = PyObject(data)
    name === :o && return o
    return getproperty(o, name)
end

function Base.show(io::IO, data::Dataset)
    out = pycall(pybuiltin("str"), String, data)
    out = replace(out, "<xarray.Dataset>" => "Dataset (xarray.Dataset)")
    print(io, out)
end
function Base.show(io::IO, ::MIME"text/html", data::Dataset)
    obj = data.o
    (:_repr_html_ in propertynames(obj)) || return show(io, data)
    out = obj._repr_html_()
    out = replace(out, r"<?xarray.Dataset>?" => "Dataset (xarray.Dataset)")
    print(io, out)
end

attributes(data::Dataset) = getproperty(PyObject(data), :_attrs)

function setattribute!(data::Dataset, key, value)
    attrs = merge(attributes(data), Dict(key => value))
    setproperty!(PyObject(data), :_attrs, attrs)
    return attrs
end

"""
    convert_to_dataset(obj; group = :posterior, kwargs...) -> Dataset

Convert a supported object to a `Dataset`.

In most cases, this function calls [`convert_to_inference_data`](@ref) and returns the
corresponding `group`.
"""
function convert_to_dataset(obj; group = :posterior, kwargs...)
    group = Symbol(group)
    idata = convert_to_inference_data(obj; group = group, kwargs...)
    dataset = getproperty(idata, group)
    return dataset
end
convert_to_dataset(data::Dataset; kwargs...) = data

"""
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
function convert_to_constant_dataset(
    obj;
    coords = nothing,
    dims = nothing,
    library = nothing,
    attrs = nothing,
)
    obj = convert(Dict, obj)
    base = arviz.data.base
    coords = coords === nothing ? Dict{String,Vector}() : coords
    dims = dims === nothing ? Dict{String,Vector{String}}() : dims

    data = Dict{String,Any}()
    for (key, vals) in obj
        vals = _asarray(vals)
        val_dims = get(dims, key, nothing)
        (val_dims, val_coords) =
            base.generate_dims_coords(size(vals), key; dims = val_dims, coords = coords)
        data[string(key)] = xarray.DataArray(vals; dims = val_dims, coords = val_coords)
    end

    default_attrs = base.make_attrs()
    if library !== nothing
        default_attrs = merge(default_attrs, Dict("inference_library" => string(library)))
    end
    attrs = attrs === nothing ? default_attrs : merge(default_attrs, attrs)
    return Dataset(data_vars = data, coords = coords, attrs = attrs)
end

"""
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
function dict_to_dataset(data; library = nothing, attrs = nothing, kwargs...)
    if library !== nothing
        ldict = Dict("inference_library" => string(library))
        attrs = (attrs === nothing ? ldict : merge(attrs, ldict))
    end
    return arviz.dict_to_dataset(data; attrs = attrs, kwargs...)
end

"""
    dataset_to_dict(ds::Dataset) -> Tuple{Dict{String,Array},NamedTuple}

Convert a `Dataset` to a dictionary of `Array`s. The function also returns keyword arguments
to [`dict_to_dataset`](@ref).
"""
function dataset_to_dict(ds::Dataset)
    ds_dict = ds.to_dict()
    data_vars = ds_dict["data_vars"]
    attrs = ds_dict["attrs"]

    coords = ds_dict["coords"]
    delete!(coords, "chain")
    delete!(coords, "draw")
    coords = Dict(k => v["data"] for (k, v) in coords)

    data = Dict{String,Array}()
    dims = Dict{String,Vector{String}}()
    for (k, v) in data_vars
        data[k] = v["data"]
        dim = v["dims"][3:end]
        if !isempty(dim)
            dims[k] = [dim...]
        end
    end

    return data, (attrs = attrs, coords = coords, dims = dims)
end
