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
    ret = NamedTuple()
    for k in K
        v = flatten.(getproperty.(x, k))
        ret = merge(ret, (k => flatten(v),))
    end
    return ret
end

@doc doc"""
    from_namedtuple(posterior::NamedTuple; kwargs...) -> InferenceData
    from_namedtuple(posterior::Vector{<:NamedTuple}; kwargs...) -> InferenceData
    from_namedtuple(posterior::Matrix{<:NamedTuple}; kwargs...) -> InferenceData
    from_namedtuple(posterior::Vector{Vector{<:NamedTuple}}; kwargs...) -> InferenceData
    from_mcmcchains(
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
    + `::Vector{<:NamedTuple}`: Each element is a `NamedTuple` from a chain with
        `Array`/`MonteCarloMeasurements.Particle` values with dimensions
        `(ndraws, sizes...)`.
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
- `observed_data::Dict{String,Array}=nothing`: Observed data on which the `posterior` is
     conditional. It should only contain data which is modeled as a random variable. Keys
     are parameter names and values.
- `constant_data::Dict{String,Array}=nothing`: Model constants, data included in the model
     which is not modeled as a random variable. Keys are parameter names and values.
- `predictions_constant_data::Dict{String,Array}=nothing`: Constants relevant to the model
     predictions (i.e. new `x` values in a linear regression).
- `log_likelihood::Any=nothing`: Pointwise log-likelihood for the data. It is recommended
     to use this argument as a dictionary whose keys are observed variable names and whose
     values are log likelihood arrays.
- `library=nothing`: Name of library that generated the draws
- `coords::Dict{String,Vector}=nothing`: Map from named dimension to named indices
- `dims::Dict{String,Vector{String}}=nothing`: Map from variable name to names of its
     dimensions

# Returns

- `InferenceData`: The data with groups corresponding to the provided data

# Examples

```@example
using ArviZ
nchains, ndraws = 2, 10

data1 = (
    x = rand(nchains, ndraws),
    y = randn(nchains, ndraws, 2),
    z = randn(nchains, ndraws, 3, 2),
)
idata1 = from_namedtuple(data1)

data2 = [(x = rand(ndraws), y = randn(ndraws, 2), z = randn(ndraws, 3, 2)) for _ = 1:nchains];
idata2 = from_namedtuple(data2)

data3 = [(x = rand(), y = randn(2), z = randn(3, 2)) for _ = 1:nchains, _ = 1:ndraws];
idata3 = from_namedtuple(data3)

data4 = [[(x = rand(), y = randn(2), z = randn(3, 2)) for _ = 1:ndraws] for _ = 1:nchains];
idata4 = from_namedtuple(data4)
```
"""
from_namedtuple

function from_namedtuple(
    posterior,
    posterior_predictive,
    sample_stats,
    predictions,
    log_likelihood;
    library=nothing,
    kwargs...,
)
    all_idata = InferenceData()
    post_dict = posterior === nothing ? nothing : convert(Dict, posterior)
    for (group, group_data) in [
        :posterior_predictive => posterior_predictive,
        :sample_stats => sample_stats,
        :predictions => predictions,
        :log_likelihood => log_likelihood,
    ]
        group_data === nothing && continue
        if post_dict !== nothing
            if group_data isa Union{Symbol,String}
                group_data = [Symbol(group_data)]
            end
            if group_data isa Union{AbstractVector{Symbol},NTuple{N,Symbol} where {N}}
                group_data = popsubdict!(post_dict, group_data)
            end
            isempty(group_data) && continue
        end
        group_dataset = convert_to_dataset(group_data; kwargs...)
        if library !== nothing
            setattribute!(group_dataset, "inference_library", string(library))
        end
        concat!(all_idata, InferenceData(; group => group_dataset))
    end

    (post_dict === nothing || isempty(post_dict)) && return all_idata

    group_dataset = convert_to_dataset(post_dict; kwargs...)
    if library !== nothing
        setattribute!(group_dataset, "inference_library", string(library))
    end
    concat!(all_idata, InferenceData(; posterior=group_dataset))

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
        pre_prior_idata = convert_to_inference_data(
            prior;
            posterior_predictive=prior_predictive,
            sample_stats=sample_stats_prior,
            library=library,
            kwargs...,
        )
        prior_idata = rekey(
            pre_prior_idata,
            Dict(
                :posterior => :prior,
                :posterior_predictive => :prior_predictive,
                :sample_stats => :sample_stats_prior,
            ),
        )
        concat!(all_idata, prior_idata)
    end

    for (group, group_data) in [
        :observed_data => observed_data,
        :constant_data => constant_data,
        :predictions_constant_data => predictions_constant_data,
    ]
        group_data === nothing && continue
        group_dict = convert(Dict, group_data)
        group_dataset = convert_to_constant_dataset(group_dict; library=library, kwargs...)
        concat!(all_idata, InferenceData(; group => group_dataset))
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
convert_to_inference_data(data::NamedTuple; kwargs...) = from_namedtuple(data; kwargs...)
function convert_to_inference_data(data::AbstractVector{<:NamedTuple}; kwargs...)
    return from_namedtuple(data; kwargs...)
end
function convert_to_inference_data(data::AbstractMatrix{<:NamedTuple}; kwargs...)
    return from_namedtuple(data; kwargs...)
end
function convert_to_inference_data(
    data::AbstractVector{<:AbstractVector{<:NamedTuple}}; kwargs...
)
    return from_namedtuple(data; kwargs...)
end
