import .MonteCarloMeasurements: AbstractParticles

stack(x::AbstractParticles) = Array(x)

function stack(v::AbstractArray{<:AbstractParticles})
    m = reduce(hcat, stack.(v))
    return Array(reshape(m, size(m, 1), size(v)...))
end

"""
    convert_to_inference_data(::AbstractParticles; kwargs...) -> InferenceData
    convert_to_inference_data(
        ::AbstractArray{<:AbstractParticles};
        kwargs...,
    ) -> InferenceData

Convert a single- or multi-dimensional `MonteCarloMeasurements.AbstractParticles`
to an [`InferenceData`](@ref).
"""
function convert_to_inference_data(obj::AbstractParticles; kwargs...)
    return convert_to_inference_data(stack(obj); kwargs...)
end

function convert_to_inference_data(obj::AbstractArray{<:AbstractParticles}; kwargs...)
    return convert_to_inference_data(stack(obj); kwargs...)
end
