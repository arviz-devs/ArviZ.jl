using .SampleChains: SampleChains
using .SampleChains.TupleVectors: TupleVector

# TODO: supported samples with a nested structure

function namedtuple_of_arrays(x::TupleVector{<:NamedTuple{K}}) where {K}
    return NamedTuple(k => recursive_stack(getproperty(x, k)) for k in K)
end
function namedtuple_of_arrays(chain::SampleChains.AbstractChain)
    return namedtuple_of_arrays(SampleChains.samples(chain))
end
function namedtuple_of_arrays(multichain::SampleChains.MultiChain)
    chains = SampleChains.getchains(multichain)
    return namedtuple_of_arrays(map(namedtuple_of_arrays, chains))
end

_maybe_multichain(x) = x
_maybe_multichain(x::SampleChains.MultiChain) = x
_maybe_multichain(x::SampleChains.AbstractChain) = SampleChains.MultiChain(x)
function _maybe_multichain(x::AbstractVector{<:SampleChains.AbstractChain})
    return SampleChains.MultiChain(x)
end
function _maybe_multichain(
    x::Tuple{<:SampleChains.AbstractChain,Vararg{<:SampleChains.AbstractChain}}
)
    return SampleChains.MultiChain(x...)
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
            energy=info.π,
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
    from_samplechains(
        posterior=nothing;
        prior=nothing,
        library=SampleChains,
        kwargs...,
    ) -> InferenceData

Convert SampleChains samples to an `InferenceData`.

Either `posterior` or `prior` may be a `SampleChains.AbstractChain` or
`SampleChains.MultiChain` object.

For descriptions of remaining `kwargs`, see [`from_namedtuple`](@ref).
"""
function from_samplechains(
    posterior=nothing;
    prior=nothing,
    sample_stats=nothing,
    sample_stats_prior=nothing,
    library=SampleChains,
    kwargs...,
)
    posterior_mc = _maybe_multichain(posterior)
    prior_mc = _maybe_multichain(prior)
    if sample_stats === nothing && posterior_mc isa SampleChains.MultiChain
        sample_stats = _samplechains_info(posterior_mc)
    end
    if sample_stats === nothing && prior_mc isa SampleChains.MultiChain
        sample_stats_prior = _samplechains_info(prior_mc)
    end
    return from_namedtuple(
        posterior_mc; prior=prior_mc, sample_stats, sample_stats_prior, library, kwargs...
    )
end

"""
    convert_to_inference_data(
        obj::SampleChains.AbstractChain;
        group=:posterior,
        kwargs...,
    ) -> InferenceData
    convert_to_inference_data(
        obj::SampleChains.AbstractChain;
        group=:posterior,
        kwargs...,
    ) -> InferenceData

Convert the chains `obj` to an [`InferenceData`](@ref) with the specified `group`.

Remaining `kwargs` are forwarded to [`from_samplechains`](@ref).
"""
function convert_to_inference_data(
    chain::T; group=:posterior, kwargs...
) where {
    T<:Union{
        SampleChains.AbstractChain,
        SampleChains.MultiChain,
        AbstractVector{<:SampleChains.AbstractChain},
        Tuple{<:SampleChains.AbstractChain,Vararg{<:SampleChains.AbstractChain}},
    },
}
    group = Symbol(group)
    group === :posterior && return from_samplechains(chain; kwargs...)
    return from_samplechains(; group => chain, kwargs...)
end
