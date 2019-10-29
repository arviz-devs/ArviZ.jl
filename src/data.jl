struct InferenceData
    o::PyObject
end

function InferenceData(args...; kwargs...)
    data = arviz.InferenceData(args...; kwargs...)
    return InferenceData(data)
end

InferenceData(data::InferenceData) = data

function Base.getproperty(data::InferenceData, name::Symbol)
    if name === :o
        return getfield(data, name)
    else
        return getproperty(data.o, name)
    end
end

function Base.show(io::IO, data::InferenceData)
    out = pycall(pybuiltin("str"), String, data.o)
    out = replace(out, r"Inference data" => "InferenceData")
    print(io, out)
end

convert_to_arviz_data(data) = data
convert_to_arviz_data(data::InferenceData) = data.o
convert_to_arviz_data(data...) = convert_to_arviz_data.(data)

"""
    convert_to_inference_data(obj; kwargs...)

Convert a supported object to an `InferenceData`.

This function sends `obj` to the right conversion function. It is idempotent,
in that it will return `InferenceData` objects unchanged.
"""
function convert_to_inference_data(obj; kwargs...)
    obj = convert_to_arviz_data(obj)
    data = arviz.convert_to_inference_data(obj; kwargs...)
    return InferenceData(data)
end

convert_to_inference_data(obj::InferenceData) = obj

function concat(args...; kwargs...)
    data = arviz.concat(convert_to_arviz_data.(args)...; kwargs...)
    return InferenceData(data)
end

function Base.:+(data1::InferenceData, data2::InferenceData)
    return InferenceData(data1.o + data2.o)
end

function from_dict(args...; kwargs...)
    data = arviz.from_dict(args...; kwargs...)
    return InferenceData(data)
end
