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

Base.display(data::InferenceData) = Base.display(data.o)

convert_to_arviz_data(data) = data
convert_to_arviz_data(data::InferenceData) = data.o
convert_to_arviz_data(data...) = convert_to_arviz_data.(data)

function convert_to_inference_data(args...; kwargs...)
    data = arviz.convert_to_inference_data(args...; kwargs...)
    return InferenceData(data)
end

convert_to_inference_data(data::InferenceData) = data

function concat(args...; kwargs...)
    data = arviz.concat(convert_to_arviz_data.(args)...; kwargs...)
    return InferenceData(data)
end

function Base.:+(data1::InferenceData, data2::InferenceData)
    return InferenceData(data1.o + data2.o)
end
