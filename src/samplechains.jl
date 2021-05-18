using .SampleChains: SampleChains

# TODO: supported samples with a nested structure

function from_samplechains(posterior::SampleChains.AbstractChain; kwargs...)
    return from_samplechains(SampleChains.MultiChain(posterior); kwargs...)
end

function from_samplechains(posterior::SampleChains.MultiChain; kwargs...)
    chains = SampleChains.getchains(posterior)
    # TODO: overload namedtuple_of_arrays for AbstractChain to be more efficient
    post_data = map(namedtuple_of_arrays, chains)
    stats_data = map(_samplechains_info, chains)
    if all(isnothing, stats_data)
        return convert_to_inference_data(post_data; kwargs...)
    else
        return convert_to_inference_data(post_data; sample_stats=stats_data, kwargs...)
    end
end

# info(::AbstractChain) is only required to return an AbstractVector, which is not enough
# information for us to convert it
# see https://github.com/arviz-devs/ArviZ.jl/issues/124
_samplechains_info(::SampleChains.AbstractChain) = nothing

@require SampleChainsDynamicHMC = "6d9fd711-e8b2-4778-9c70-c1dfb499d4c4" begin
    using .SampleChainsDynamicHMC: SampleChainsDynamicHMC

    function _samplechains_info(chain::SampleChainsDynamicHMC.DynamicHMCChain)
        info = SampleChains.info(chain)
        termination = info.termination
        tree_stats = (
            lp=info.π,
            tree_depth=info.depth,
            acceptance_rate=info.acceptance_rate,
            n_steps=info.steps,
            diverging=map(t -> t.left == t.right, termination),
            turning=map(t -> t.left < t.right, termination),
        )
        used_info = (:π, :depth, :acceptance_rate, :steps, :termination)
        skipped_info = setdiff(propertynames(info), used_info)
        isempty(skipped_info) ||
            @debug "Skipped SampleChainsDynamicHMC info entries: $skipped_info."
        return tree_stats
    end
end

"""
    convert_to_inference_data(obj::SampleChains.AbstractChain; group = :posterior, kwargs...) -> InferenceData

Convert the chains `obj` to an [`InferenceData`](@ref) with the specified `group`.

Remaining `kwargs` are forwarded to [`from_mcmcchains`](@ref).
"""
function convert_to_inference_data(
    chain::T; group=:posterior, kwargs...
) where {T<:Union{SampleChains.AbstractChain,SampleChains.MultiChain}}
    group = Symbol(group)
    group === :posterior && return from_samplechains(chain; kwargs...)
    return from_samplechains(; group => chain)
end
