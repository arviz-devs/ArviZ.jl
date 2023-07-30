module ArviZSampleChainsExt

using .SampleChains: SampleChains
using .SampleChains.TupleVectors: TupleVector

# TODO: supported samples with a nested structure

function namedtuple_of_arrays(chain::SampleChains.AbstractChain)
    return InferenceObjects.stack_draws(SampleChains.samples(chain))
end
function namedtuple_of_arrays(multichain::SampleChains.MultiChain)
    chains = SampleChains.getchains(multichain)
    return InferenceObjects.stack_chains(map(InferenceObjects.stack_draws, chains))
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
    if sample_stats_prior === nothing && prior_mc isa SampleChains.MultiChain
        sample_stats_prior = _samplechains_info(prior_mc)
    end
    posterior_nt = posterior === nothing ? nothing : namedtuple_of_arrays(posterior_mc)
    prior_nt = if prior_mc isa SampleChains.MultiChain
        namedtuple_of_arrays(prior_mc)
    else
        prior_mc
    end
    return from_namedtuple(
        posterior_nt; prior=prior_nt, sample_stats, sample_stats_prior, library, kwargs...
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

end  # module
