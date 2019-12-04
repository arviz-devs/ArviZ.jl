"""
    stack(x::NamedTuple) -> NamedTuple
    stack(x::AbstractArray{NamedTuple}) -> NamedTuple
    stack(x::AbstractArray{AbstractArray{<:NamedTuple}}) -> NamedTuple

Given a container of `NamedTuple`s, concatenate them, using the container
dimensions as the dimensions of the resulting arrays.

# Examples

```@example
using ArviZ
nchains, ndraws = 4, 100
data = [(x = rand(), y = randn(2), z = randn(2, 3)) for _ in 1:nchains, _ in 1:ndraws];
stacked_data = ArviZ.stack(data);
```
"""
stack(x) = x
stack(x::AbstractArray{T}) where {T<:Number} = Array(x)
stack(x::AbstractArray) = stack(stack.(x))
stack(x::NamedTuple) = (; (k => stack(v) for (k, v) in pairs(x))...)

function stack(x::AbstractArray{S}) where {T<:Number,N,S<:AbstractArray{T,N}}
    ret = Array{T}(undef, (size(x)..., size(x[1])...))
    @simd for k in keys(x)
        @inbounds setindex!(ret, x[k], k, (Colon() for _ = 1:N)...)
    end
    return ret
end

function stack(x::AbstractArray{<:NamedTuple{K}}) where {K}
    length(x) == 0 && return
    @inbounds x1 = x[1]
    ret = NamedTuple()
    for k in K
        v = replacemissing.(stack.(getproperty.(x, k)))
        ret = merge(ret, (k => stack(v),))
    end
    return ret
end

"""
    from_namedtuple(posterior::NamedTuple; kwargs...) -> InferenceData
    from_namedtuple(posterior::Vector{<:NamedTuple}; kwargs...) -> InferenceData
    from_namedtuple(posterior::Matrix{<:NamedTuple}; kwargs...) -> InferenceData
    from_namedtuple(posterior::Vector{Vector{<:NamedTuple}}; kwargs...) -> InferenceData

Convert a `NamedTuple` or container of `NamedTuple`s to an `InferenceData`.

If containers are passed, they are flattened into a single `NamedTuple` with
array elements whose first dimensions correspond to the dimensions of the
containers.

# Arguments

- `posterior`: The data to be converted. It may be of the following types:
    + `::NamedTuple`: The keys are the variable names and the values are arrays
        with dimensions `(nchains, ndraws, sizes...)`.
    + `::Vector{<:NamedTuple}`: Each element is a `NamedTuple` from a chain
        with array values with dimensions `(ndraws, sizes...)`.
    + `::Matrix{<:NamedTuple}`: Each element is a single draw from a single
        chain, with array/scalar values with dimensions `sizes`. The dimensions
        of the matrix container are `(nchains, ndraws)`
    + `::Vector{Vector{<:NamedTuple}}`: The same as the above case.

# Keywords

- `posterior_predictive::Any=nothing`: Draws from the posterior predictive
     distribution
- `sample_stats::Any=nothing`: Statistics of the posterior sampling process
- `prior::Any=nothing`: Draws from the prior
- `prior_predictive::Any=nothing`: Draws from the prior predictive distribution
- `sample_stats_prior::Any=nothing`: Statistics of the prior sampling process
- `observed_data::Dict{String,Array}=nothing`: Observed data on which the
     `posterior` is conditional. It should only contain data which is modeled as
     a random variable. Keys are parameter names and values.
- `constant_data::Dict{String,Array}=nothing`: Model constants, data included
     in the model which is not modeled as a random variable. Keys are parameter
     names and values.
- `library=nothing`: Name of library that generated the draws
- `coords::Dict{String,Vector}=nothing`: Map from named dimension to named
     indices
- `dims::Dict{String,Vector{String}}=nothing`: Map from variable name to names
     of its dimensions

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
function from_namedtuple(
    posterior::Union{NamedTuple,Nothing} = nothing;
    posterior_predictive = nothing,
    sample_stats = nothing,
    prior = nothing,
    prior_predictive = nothing,
    sample_stats_prior = nothing,
    observed_data = nothing,
    constant_data = nothing,
    library = nothing,
    kwargs...,
)
    group_datasets = Dict{Symbol,Dataset}()

    for (group, group_data) in [
        :posterior => posterior,
        :posterior_predictive => posterior_predictive,
        :sample_stats => sample_stats,
        :prior => prior,
        :prior_predictive => prior_predictive,
        :sample_stats_prior => sample_stats_prior,
    ]
        if group_data !== nothing
            if group === :posterior
                group_data = convert(Dict, group_data)
            end
            group_dataset = convert_to_dataset(group_data; kwargs...)
            if library !== nothing
                setattribute!(group_dataset, "inference_library", string(library))
            end
            group_datasets[group] = group_dataset
        end
    end

    for (group, group_data) in [
        :observed_data => observed_data,
        :constant_data => constant_data,
    ]
        if group_data !== nothing
            group_dict = convert(Dict, group_data)
            group_datasets[group] = convert_to_constant_dataset(
                group_dict;
                library = library,
                kwargs...,
            )
        end
    end

    return InferenceData(; group_datasets...)
end

from_namedtuple(data::AbstractVector{<:NamedTuple}; kwargs...) =
    from_namedtuple(stack(data); kwargs...)
from_namedtuple(data::AbstractMatrix{<:NamedTuple}; kwargs...) =
    from_namedtuple(stack(data); kwargs...)
from_namedtuple(data::AbstractVector{<:AbstractVector{<:NamedTuple}}; kwargs...) =
    from_namedtuple(stack(data); kwargs...)

"""
    convert_to_inference_data(obj::NamedTuple; kwargs...) -> InferenceData
    convert_to_inference_data(obj::Vector{<:NamedTuple}; kwargs...) -> InferenceData
    convert_to_inference_data(obj::Matrix{<:NamedTuple}; kwargs...) -> InferenceData
    convert_to_inference_data(obj::Vector{Vector{<:NamedTuple}}; kwargs...) -> InferenceData

Convert `obj` to an [`InferenceData`](@ref). See [`from_namedtuple`](@ref) for
a description of `obj` possibilities and `kwargs`.
"""
convert_to_inference_data(data::NamedTuple; kwargs...) = from_namedtuple(data; kwargs...)
convert_to_inference_data(data::AbstractVector{<:NamedTuple}; kwargs...) =
    from_namedtuple(stack(data); kwargs...)
convert_to_inference_data(data::AbstractMatrix{<:NamedTuple}; kwargs...) =
    from_namedtuple(stack(data); kwargs...)
convert_to_inference_data(data::AbstractVector{<:AbstractVector{<:NamedTuple}}; kwargs...) =
    from_namedtuple(stack(data); kwargs...)
