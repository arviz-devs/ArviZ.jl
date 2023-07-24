const HDI_DEFAULT_PROB = 0.94
const HDI_BOUND_DIM = Dimensions.Dim{:hdi_bound}([:lower, :upper])

"""
    hdi(samples::AbstractArray{<:Real}; prob=$(HDI_DEFAULT_PROB)) -> (; lower, upper)

Calculate the unimodal highest density interval (HDI) of `samples` for the probability
`prob`.[^Hyndman1996]

The HDI is the minimum width Bayesian credible interval (BCI). That is, it is the smallest
possible interval containing `(100*prob)`% of the draws.[^Hyndman1996]

`samples` is an array of shape `(draws[, chains[, params...]])`. If multiple parameters are
present, then `lower` and `upper` are arrays with the shape `(params...,)`, computed
separately for each marginal.

!!! note
    Any default value of `prob` is arbitrary. The default value of
    `prob=$(HDI_DEFAULT_PROB)` instead of a more common default like `prob=0.95` is chosen
    to reminder the user of this arbitrariness.

[^Hyndman1996]: Rob J. Hyndman. Computing and Graphing Highest Density Regions. (1996).
                Amer. Stat., 50(2): 120-6.
                doi: [10.1080/00031305.1996.10474359](https://doi.org/10.1080/00031305.1996.10474359)
                [jstor](https://doi.org/10.2307/2684423).

# Examples

Here we calculate the 83% HDI for a normal random variable

```jldoctest; setup = :(using Random; Random.seed!(78))
x = randn(2_000)
hdi(x; prob=0.83)

# output

(lower = -1.3826605224220527, upper = 1.2580552089709671)
```

```jldoctest; setup = :(using Random; Random.seed!(67))
x = randn(1_000, 1, 1) .+ reshape(0:5:10, 1, 1, :)
pairs(hdi(x))

# output

pairs(::NamedTuple) with 2 entries:
  :lower => [-1.9505, 3.0495, 8.0495]
  :upper => [1.90028, 6.90028, 11.9003]
```
"""
function hdi(x::AbstractArray{<:Real}; kwargs...)
    xcopy = similar(x)
    copyto!(xcopy, x)
    return hdi!(xcopy; kwargs...)
end
function hdi!(x::AbstractArray{<:Real}; prob::Real=HDI_DEFAULT_PROB)
    0 < prob ≤ 1 || throw(ArgumentError("HDI `prob` must be between 0 and 1."))
    return _hdi!(x, prob)
end

function _hdi!(x::AbstractVecOrMat{<:Real}, prob::Real)
    xvec = vec(x)
    n = length(xvec)
    tail_length = max(1, ceil(Int, (1 - prob) * n))
    # for prob ⪆ 0.7, performing 2 partialsorts is faster than sorting the whole array
    lower_tail = partialsort!(xvec, 1:tail_length)
    upper_tail = partialsort!(xvec, (n - tail_length + 1):n)
    upper, lower = argmin(Base.splat(-), zip(upper_tail, lower_tail))
    return (; lower, upper)
end
function _hdi!(x::AbstractArray{<:Real}, prob::Real)
    axes_out = _param_axes(x)
    lower = similar(x, axes_out)
    upper = similar(x, axes_out)
    for (i, x_slice) in zip(eachindex(lower), _eachparam(x))
        lower[i], upper[i] = _hdi!(x_slice, prob)
    end
    return (; lower, upper)
end

"""
    hdi(data::InferenceData; kwargs...) -> Dataset
    hdi(data::Dataset; kwargs...) -> Dataset

Calculate the highest density interval (HDI) for each parameter in the data.

# Example

Calculate HDI for all parameters in the `posterior` group of an `InferenceData`:

```jldoctest hdi_infdata
using ArviZ, ArviZExampleData
idata = load_example_data("centered_eight")
hdi(idata)

# output

Dataset with dimensions:
  Dim{:hdi_bound} Categorical{Symbol} Symbol[:lower, :upper] ForwardOrdered,
  Dim{:school} Categorical{String} String[Choate, Deerfield, …, St. Paul's, Mt. Hermon] Unordered
and 3 layers:
  :mu    Float64 dims: Dim{:hdi_bound} (2)
  :theta Float64 dims: Dim{:school}, Dim{:hdi_bound} (8×2)
  :tau   Float64 dims: Dim{:hdi_bound} (2)
```

We can also calculate the HDI for a subset of variables:

```jldoctest hdi_infdata
julia> hdi(idata.posterior[(:mu, :theta)])
Dataset with dimensions:
  Dim{:hdi_bound} Categorical{Symbol} Symbol[:lower, :upper] ForwardOrdered,
  Dim{:school} Categorical{String} String[Choate, Deerfield, …, St. Paul's, Mt. Hermon] Unordered
and 2 layers:
  :mu    Float64 dims: Dim{:hdi_bound} (2)
  :theta Float64 dims: Dim{:school}, Dim{:hdi_bound} (8×2)
```
"""
hdi(data::InferenceObjects.InferenceData; kwargs...) = hdi(data.posterior; kwargs...)
function hdi(data::InferenceObjects.Dataset; kwargs...)
    ds = map(DimensionalData.layers(data)) do var
        lower, upper = hdi(_draw_chains_params_array(var); kwargs...)
        return cat(_as_dimarray(lower, var), _as_dimarray(upper, var); dims=HDI_BOUND_DIM)
    end
    return InferenceObjects.Dataset(ds)
end
