import .MonteCarloMeasurements: AbstractParticles

stack(x::AbstractParticles) = Array(x)

function stack(v::AbstractArray{<:AbstractParticles})
    m = reduce(hcat, stack.(v))
    return Array(reshape(m, size(m, 1), size(v)...))
end

"""
    convert_to_inference_data(obj::AbstractParticles; kwargs...) -> InferenceData
    convert_to_inference_data(
        obj::AbstractVector{<:AbstractParticles};
        kwargs...,
    ) -> InferenceData
    convert_to_inference_data(
        obj::AbstractVector{<:AbstractArray{<:AbstractParticles}};
        kwargs...,
    ) -> InferenceData

Convert `MonteCarloMeasurements.AbstractParticles` to an [`InferenceData`](@ref).

`obj` may have the following types:
- `::AbstractParticles`: Univariate draws from a single chain.
- `::AbstractVector{<:AbstractParticles}`: Univariate draws from a vector of
     chains.
- `::AbstractVector{<:AbstractArray{<:AbstractParticles}}`: Multivariate
     draws from a vector of chains.
"""
function convert_to_inference_data(obj::AbstractParticles; kwargs...)
    return convert_to_inference_data([obj]; kwargs...)
end

function convert_to_inference_data(obj::AbstractVector{<:AbstractParticles}; kwargs...)
    return convert_to_inference_data(stack(stack.(obj)); kwargs...)
end

function convert_to_inference_data(
    obj::AbstractVector{<:AbstractArray{<:AbstractParticles}};
    kwargs...,
)
    return convert_to_inference_data(stack(stack.(obj)); kwargs...)
end
