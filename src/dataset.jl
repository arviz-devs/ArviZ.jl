@forwardfun convert_to_dataset
@forwardfun dict_to_dataset

"""
    dataset_to_dict(ds::PyObject)

Convert an `xarray.Dataset` to a dictionary of `Array`s. The function is the
returns a `Dict{String,Array}` and a `NamedTuple` with keyword arguments to
`dict_to_dataset`.
"""
function dataset_to_dict(ds::PyObject)
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
