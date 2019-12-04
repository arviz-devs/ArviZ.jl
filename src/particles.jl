import .MonteCarloMeasurements: AbstractParticles

stack(x::AbstractParticles) = Array(x)

function stack(v::AbstractArray{<:AbstractParticles})
    m = reduce(hcat, stack.(v))
    return Array(reshape(m, size(m, 1), size(v)...))
end

function convert_to_inference_data(data::AbstractParticles; kwargs...)
    return convert_to_inference_data(stack(data); kwargs...)
end

function convert_to_inference_data(data::AbstractArray{<:AbstractParticles}; kwargs...)
    return convert_to_inference_data(stack(data); kwargs...)
end
