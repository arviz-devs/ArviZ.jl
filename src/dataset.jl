"""
    Dataset(::PyObject)

Loose wrapper around `xarray.Dataset`, mostly used for dispatch.

To create a `Dataset`, use [`convert_to_dataset`](@ref).
"""
struct Dataset
    o::PyObject

    function Dataset(o::PyObject)
        pyisinstance(
            o,
            xarray.Dataset,
        ) || throw(ArgumentError("$o is not an `xarray.Dataset`."))
        return new(o)
    end
end

@inline Dataset(data::Dataset) = data

@inline PyObject(data::Dataset) = getfield(data, :o)

Base.convert(::Type{Dataset}, o::PyObject) = Dataset(o)

Base.hash(data::Dataset) = hash(PyObject(data))

Base.propertynames(data::Dataset) = propertynames(PyObject(data))

function Base.getproperty(data::Dataset, name::Symbol)
    o = PyObject(data)
    name === :o && return o
    return getproperty(o, name)
end

function Base.show(io::IO, data::Dataset)
    out = pycall(pybuiltin("str"), String, data)
    out = replace(out, "<xarray.Dataset>" => "Dataset")
    print(io, out)
end

"""
    convert_to_dataset(obj; group = :posterior, kwargs...) -> Dataset

Convert a supported object to a `Dataset`. In most cases, this function calls
[`convert_to_inference_data`](@ref) and returns the corresponding `group`.
"""
function convert_to_dataset(obj; group = :posterior, kwargs...)
    group = Symbol(group)
    idata = convert_to_inference_data(obj; group = group, kwargs...)
    dataset = getproperty(idata, group)
    return dataset
end

convert_to_dataset(data::Dataset; kwargs...) = data

"""
    dict_to_dataset(data::Dict{String,Array}; kwargs...) -> Dataset

Convert a dictionary with data and keys as variable names to a [`Dataset`](@ref).

# Keywords
- `attrs::Dict{String,Any}`: Json serializable metadata to attach to the
    dataset, in addition to defaults.
- `library::String`: Name of library used for performing inference. Will be
    attached to the attrs metadata.
- `coords::Dict{String,Array}`: Coordinates for the dataset
- `dims::Dict{String,Vector{String}}`: Dimensions of each variable. The keys
    are variable names, values are vectors of coordinates.

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

Convert a `Dataset` to a dictionary of `Array`s. The function also
returns keyword arguments to [`dict_to_dataset`](@ref).
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
