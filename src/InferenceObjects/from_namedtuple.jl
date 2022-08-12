"""
    namedtuple_of_arrays(x::NamedTuple) -> NamedTuple
    namedtuple_of_arrays(x::AbstractArray{NamedTuple}) -> NamedTuple
    namedtuple_of_arrays(x::AbstractArray{AbstractArray{<:NamedTuple}}) -> NamedTuple

Given a container of `NamedTuple`s, concatenate them, using the container dimensions as the
dimensions of the resulting arrays.

# Examples

```@example
using ArviZ
nchains, ndraws = 4, 100
data = [(x=rand(), y=randn(2), z=randn(2, 3)) for _ in 1:nchains, _ in 1:ndraws];
ntarray = ArviZ.namedtuple_of_arrays(data);
```
"""
namedtuple_of_arrays(x::NamedTuple) = map(flatten, x)
namedtuple_of_arrays(x::AbstractArray) = namedtuple_of_arrays(namedtuple_of_arrays.(x))
function namedtuple_of_arrays(x::AbstractArray{<:NamedTuple{K}}) where {K}
    return mapreduce(merge, K) do k
        v = flatten.(getproperty.(x, k))
        return (; k => flatten(v))
    end
end

"""
    from_namedtuple(posterior::NamedTuple; kwargs...) -> InferenceData
    from_namedtuple(posterior::Vector{<:NamedTuple}; kwargs...) -> InferenceData
    from_namedtuple(posterior::Matrix{<:NamedTuple}; kwargs...) -> InferenceData
    from_namedtuple(posterior::Vector{Vector{<:NamedTuple}}; kwargs...) -> InferenceData
    from_namedtuple(
        posterior::NamedTuple,
        sample_stats::Any,
        posterior_predictive::Any,
        predictions::Any,
        log_likelihood::Any;
        kwargs...
    ) -> InferenceData

Convert a `NamedTuple` or container of `NamedTuple`s to an `InferenceData`.

If containers are passed, they are flattened into a single `NamedTuple` with array elements
whose first dimensions correspond to the dimensions of the containers.

# Arguments

  - `posterior`: The data to be converted. It may be of the following types:
    
      + `::NamedTuple`: The keys are the variable names and the values are arrays with
        dimensions `(nchains, ndraws, sizes...)`.
      + `::Matrix{<:NamedTuple}`: Each element is a single draw from a single chain, with
        array/scalar values with dimensions `sizes`. The dimensions of the matrix container
        are `(nchains, ndraws)`
      + `::Vector{Vector{<:NamedTuple}}`: The same as the above case.

# Keywords

  - `posterior_predictive::Any=nothing`: Draws from the posterior predictive distribution
  - `sample_stats::Any=nothing`: Statistics of the posterior sampling process
  - `predictions::Any=nothing`: Out-of-sample predictions for the posterior.
  - `prior::Any=nothing`: Draws from the prior
  - `prior_predictive::Any=nothing`: Draws from the prior predictive distribution
  - `sample_stats_prior::Any=nothing`: Statistics of the prior sampling process
  - `observed_data::NamedTuple`: Observed data on which the `posterior` is
    conditional. It should only contain data which is modeled as a random variable. Keys
    are parameter names and values.
  - `constant_data::NamedTuple`: Model constants, data included in the model
    which is not modeled as a random variable. Keys are parameter names and values.
  - `predictions_constant_data::NamedTuple`: Constants relevant to the model
    predictions (i.e. new `x` values in a linear regression).
  - `log_likelihood`: Pointwise log-likelihood for the data. It is recommended
    to use this argument as a `NamedTuple` whose keys are observed variable names and whose
    values are log likelihood arrays.
  - `library`: Name of library that generated the draws
  - `coords`: Map from named dimension to named indices
  - `dims`: Map from variable name to names of its dimensions

# Returns

  - `InferenceData`: The data with groups corresponding to the provided data

# Examples

```@example
using ArviZ
nchains, ndraws = 2, 10

data1 = (
    x=rand(nchains, ndraws), y=randn(nchains, ndraws, 2), z=randn(nchains, ndraws, 3, 2)
)
idata1 = from_namedtuple(data1)

data2 = [(x=rand(ndraws), y=randn(ndraws, 2), z=randn(ndraws, 3, 2)) for _ in 1:nchains];
idata2 = from_namedtuple(data2)

data3 = [(x=rand(), y=randn(2), z=randn(3, 2)) for _ in 1:nchains, _ in 1:ndraws];
idata3 = from_namedtuple(data3)

data4 = [[(x=rand(), y=randn(2), z=randn(3, 2)) for _ in 1:ndraws] for _ in 1:nchains];
idata4 = from_namedtuple(data4)
```
"""
from_namedtuple

function from_namedtuple(
    posterior, posterior_predictive, sample_stats, predictions, log_likelihood; kwargs...
)
    all_idata = InferenceData()
    post_data = posterior === nothing ? posterior : namedtuple_of_arrays(posterior)
    for (group, group_data) in [
        :posterior_predictive => posterior_predictive,
        :sample_stats => sample_stats,
        :predictions => predictions,
        :log_likelihood => log_likelihood,
    ]
        group_data === nothing && continue
        if post_data !== nothing
            if group_data isa Symbol
                group_data = (Symbol(group_data),)
            end
            if all(Base.Fix2(isa, Symbol), group_data)
                group_data = NamedTuple{Tuple(group_data)}(post_data)
                post_data = NamedTuple{Tuple(setdiff(keys(post_data), keys(group_data)))}(
                    post_data
                )
            end
            isempty(group_data) && continue
        end
        group_dataset = convert_to_dataset(group_data; kwargs...)
        all_idata = merge(all_idata, InferenceData(; group => group_dataset))
    end

    (post_data === nothing || isempty(post_data)) && return all_idata

    post_dataset = convert_to_dataset(post_data; kwargs...)
    all_idata = merge(all_idata, InferenceData(; posterior=post_dataset))

    return all_idata
end
function from_namedtuple(
    posterior::Union{NamedTuple,Nothing}=nothing;
    posterior_predictive=nothing,
    sample_stats=nothing,
    predictions=nothing,
    prior=nothing,
    prior_predictive=nothing,
    sample_stats_prior=nothing,
    observed_data=nothing,
    constant_data=nothing,
    predictions_constant_data=nothing,
    log_likelihood=nothing,
    library=nothing,
    kwargs...,
)
    all_idata = from_namedtuple(
        posterior,
        posterior_predictive,
        sample_stats,
        predictions,
        log_likelihood;
        library=library,
        kwargs...,
    )

    if any(x -> x !== nothing, [prior, prior_predictive, sample_stats_prior])
        pre_prior_idata = from_namedtuple(
            prior;
            posterior_predictive=prior_predictive,
            sample_stats=sample_stats_prior,
            library,
            kwargs...,
        )
        prior_idata = rekey(
            pre_prior_idata,
            (
                posterior=:prior,
                posterior_predictive=:prior_predictive,
                sample_stats=:sample_stats_prior,
            ),
        )
        all_idata = merge(all_idata, prior_idata)
    end

    for (group, group_data) in [
        :observed_data => observed_data,
        :constant_data => constant_data,
        :predictions_constant_data => predictions_constant_data,
    ]
        group_data === nothing && continue
        group_dataset = convert_to_dataset(group_data; library, default_dims=(), kwargs...)
        all_idata = merge(all_idata, InferenceData(; group => group_dataset))
    end

    return all_idata
end
function from_namedtuple(data::AbstractVector{<:NamedTuple}; kwargs...)
    return from_namedtuple(namedtuple_of_arrays(data); kwargs...)
end
function from_namedtuple(data::AbstractMatrix{<:NamedTuple}; kwargs...)
    return from_namedtuple(namedtuple_of_arrays(data); kwargs...)
end
function from_namedtuple(data::AbstractVector{<:AbstractVector{<:NamedTuple}}; kwargs...)
    return from_namedtuple(namedtuple_of_arrays(data); kwargs...)
end

"""
    convert_to_inference_data(obj::NamedTuple; kwargs...) -> InferenceData
    convert_to_inference_data(obj::Vector{<:NamedTuple}; kwargs...) -> InferenceData
    convert_to_inference_data(obj::Matrix{<:NamedTuple}; kwargs...) -> InferenceData
    convert_to_inference_data(obj::Vector{Vector{<:NamedTuple}}; kwargs...) -> InferenceData

Convert `obj` to an [`InferenceData`](@ref). See [`from_namedtuple`](@ref) for a description
of `obj` possibilities and `kwargs`.
"""
function convert_to_inference_data(
    data::T; group=:posterior, kwargs...
) where {
    T<:Union{
        NamedTuple,
        AbstractVector{<:NamedTuple},
        AbstractMatrix{<:NamedTuple},
        AbstractVector{<:AbstractVector{<:NamedTuple}},
    },
}
    group = Symbol(group)
    group === :posterior && return from_namedtuple(data; kwargs...)
    return from_namedtuple(; group => data, kwargs...)
end
