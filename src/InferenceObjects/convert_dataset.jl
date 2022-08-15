Base.convert(::Type{Dataset}, obj) = convert_to_dataset(obj)
Base.convert(::Type{Dataset}, obj::Dataset) = obj

"""
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
