"""
    InferenceData

Loose wrapper around `arviz.InferenceData`, which is a container for inference
data storage using xarray.

# Constructor

    InferenceData(o::PyObject)

wraps an `arviz.InferenceData`. To create an `InferenceData`, use the exported
`from_xyz` functions.
"""
struct InferenceData
    o::PyObject

    function InferenceData(o::PyObject)
        pyisinstance(
            o,
            arviz.InferenceData,
        ) || raise(ArgumentError("$o is not an `arviz.InferenceData`."))
        return new(o)
    end
end

@inline InferenceData(data::InferenceData) = data

@inline PyObject(data::InferenceData) = getfield(data, :o)

Base.convert(::Type{InferenceData}, o::PyObject) = InferenceData(o)

Base.hash(data::InferenceData) = hash(PyObject(data))

Base.propertynames(data::InferenceData) = propertynames(PyObject(data))

function Base.getproperty(data::InferenceData, name::Symbol)
    name === :o && return getfield(data, name)
    return getproperty(data.o, name)
end

Base.delete!(data::InferenceData, name) = PyObject(data).__delattr__(string(name))

(data1::InferenceData + data2::InferenceData) = data1.o + data2.o

function Base.show(io::IO, data::InferenceData)
    out = pycall(pybuiltin("str"), String, data)
    out = replace(out, r"Inference data" => "InferenceData")
    print(io, out)
end

"""
    convert_to_inference_data(obj; kwargs...)

Convert a supported object to an `InferenceData`.

This function sends `obj` to the right conversion function. It is idempotent,
in that it will return `InferenceData` objects unchanged.
"""
@forwardfun convert_to_inference_data

@inline convert_to_inference_data(obj::InferenceData) = obj

"""
    concat(args::InferenceData...; copy = true, reset_dim = true)

Concatenate `InferenceData` objects.

Concatenates over `group`, `chain` or `draw`. By default concatenates over
unique groups. To concatenate over `chain` or `draw` function needs identical
groups and variables.

The `variables` in the `data` -group are merged if `dim` are not found.

# Keyword Arguments
- dim::String If defined, concatenated over the defined dimension.
              Dimension which is concatenated. If `nothing`, concatenates over
              unique groups.
- copy::Bool If `true`, groups are copied to the new `InferenceData` object.
             Used only if `dim` is `nothing`.
- inplace::Bool If `true`, merge `args` to first object.
- reset_dim::Bool Valid only if `dim` is not `nothing`.
"""
@forwardfun concat

@forwardfun from_dict
