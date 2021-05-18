using .SampleChains: SampleChains
using .SampleChains.TupleVectors: TupleVector

# TODO: supported samples with a nested structure

function namedtuple_of_arrays(x::TupleVector{<:NamedTuple{K}}) where {K}
    return NamedTuple{K}(getproperty.(Ref(x), K))
end
function namedtuple_of_arrays(chain::SampleChains.AbstractChain)
    return namedtuple_of_arrays(SampleChains.samples(chain))
end
function namedtuple_of_arrays(multichain::SampleChains.MultiChain)
    chains = SampleChains.getchains(multichain)
    return namedtuple_of_arrays(map(namedtuple_of_arrays, chains))
end

function from_samplechains(
    posterior=nothing;
    prior=nothing,
    sample_stats=nothing,
    sample_stats_prior=nothing,
    library=:SampleChains,
    kwargs...,
)
    if sample_stats === nothing &&
       posterior isa Union{SampleChains.AbstractChain,SampleChains.MultiChain}
        sample_stats = _samplechains_info(posterior)
    end
    if sample_stats_prior === nothing &&
       prior isa Union{SampleChains.AbstractChain,SampleChains.MultiChain}
        sample_stats_prior = _samplechains_info(prior)
    end
    return from_namedtuple(
        posterior;
        prior=prior,
        sample_stats=sample_stats,
        sample_stats_prior=sample_stats_prior,
        library=library,
        kwargs...,
    )
end

# info(::AbstractChain) is only required to return an AbstractVector, which is not enough
# information for us to convert it
# see https://github.com/arviz-devs/ArviZ.jl/issues/124
_samplechains_info(::SampleChains.AbstractChain) = nothing
function _samplechains_info(multichain::SampleChains.MultiChain)
    stats = map(_samplechains_info, SampleChains.getchains(multichain))
    all(isnothing, stats) && return nothing
    return namedtuple_of_arrays(stats)
end

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
