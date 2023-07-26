const HDI_DEFAULT_PROB = 0.94
# this pattern ensures that the type is completely specified at compile time
const HDI_BOUND_DIM = Dimensions.format(
    Dimensions.Dim{:hdi_bound}([:lower, :upper]), Base.OneTo(2)
)

"""
    hdi(samples::AbstractArray{<:Real}; prob=$(HDI_DEFAULT_PROB)) -> (; lower, upper)

Estimate the unimodal highest density interval (HDI) of `samples` for the probability `prob`.

The HDI is the minimum width Bayesian credible interval (BCI). That is, it is the smallest
possible interval containing at least `(100*prob)`% of the draws.[^Hyndman1996]

`samples` is an array of shape `(draws[, chains[, params...]])`. If multiple parameters are
present, then `lower` and `upper` are arrays with the shape `(params...,)`, computed
separately for each marginal.

This implementation uses the algorithm of [^ChenShao1999].

!!! note
    Any default value of `prob` is arbitrary. The default value of
    `prob=$(HDI_DEFAULT_PROB)` instead of a more common default like `prob=0.95` is chosen
    to reminder the user of this arbitrariness.

[^Hyndman1996]: Rob J. Hyndman (1996) Computing and Graphing Highest Density Regions,
                Amer. Stat., 50(2): 120-6.
                DOI: [10.1080/00031305.1996.10474359](https://doi.org/10.1080/00031305.1996.10474359)
                [jstor](https://doi.org/10.2307/2684423).
[^ChenShao1999]: Ming-Hui Chen & Qi-Man Shao (1999)
                 Monte Carlo Estimation of Bayesian Credible and HPD Intervals,
                 J Comput. Graph. Stat., 8:1, 69-92.
                 DOI: [10.1080/10618600.1999.10474802](https://doi.org/10.1080/00031305.1996.10474359)
                 [jstor](https://doi.org/10.2307/1390921).

# Examples

Here we calculate the 83% HDI for a normal random variable:

```jldoctest hdi; setup = :(using Random; Random.seed!(78))
using ArviZ
x = randn(2_000)
hdi(x; prob=0.83)

# output

(lower = -1.3826605224220527, upper = 1.259817149822839)
```

We can also calculate the HDI for a 3-dimensional array of samples:

```jldoctest hdi; setup = :(using Random; Random.seed!(67))
x = randn(1_000, 1, 1) .+ reshape(0:5:10, 1, 1, :)
pairs(hdi(x))

# output

pairs(::NamedTuple) with 2 entries:
  :lower => [-1.9674, 3.0326, 8.0326]
  :upper => [1.90028, 6.90028, 11.9003]
```
"""
function hdi(x::AbstractArray{<:Real}; kwargs...)
    xcopy = similar(x)
    copyto!(xcopy, x)
    return hdi!(xcopy; kwargs...)
end

"""
    hdi!(samples::AbstractArray{<:Real}; prob=$(HDI_DEFAULT_PROB)) -> (; lower, upper)

A version of [hdi](@ref) that sorts `samples` in-place while computing the HDI.
"""
function hdi!(x::AbstractArray{<:Real}; prob::Real=HDI_DEFAULT_PROB)
    0 < prob < 1 || throw(DomainError(prob, "HDI `prob` must be in the range `(0, 1)`.]"))
    return _hdi!(x, prob)
end

function _hdi!(x::AbstractVector{<:Real}, prob::Real)
    isempty(x) && throw(ArgumentError("HDI cannot be computed for an empty array."))
    n = length(x)
    interval_length = floor(Int, prob * n) + 1
    if any(isnan, x) || interval_length == n
        lower, upper = extrema(x)
    else
        npoints_to_check = n - interval_length + 1
        sort!(x)
        lower_range = @views x[begin:(begin - 1 + npoints_to_check)]
        upper_range = @views x[(begin - 1 + interval_length):end]
        lower, upper = argmax(Base.splat(-), zip(lower_range, upper_range))
    end
    return (; lower, upper)
end
_hdi!(x::AbstractMatrix{<:Real}, prob::Real) = _hdi!(vec(x), prob)
function _hdi!(x::AbstractArray{<:Real}, prob::Real)
    ndims(x) > 0 ||
        throw(ArgumentError("HDI cannot be computed for a 0-dimensional array."))
    axes_out = _param_axes(x)
    lower = similar(x, axes_out)
    upper = similar(x, axes_out)
    for (i, x_slice) in zip(eachindex(lower), _eachparam(x))
        lower[i], upper[i] = _hdi!(x_slice, prob)
    end
    return (; lower, upper)
end

"""
    hdi(data::InferenceData; prob=$HDI_DEFAULT_PROB) -> Dataset
    hdi(data::Dataset; prob=$HDI_DEFAULT_PROB) -> Dataset

Calculate the highest density interval (HDI) for each parameter in the data.

# Example

Calculate HDI for all parameters in the `posterior` group of an `InferenceData`:

```jldoctest hdi_infdata
using ArviZ, ArviZExampleData
idata = load_example_data("centered_eight")
hdi(idata)

# output

Dataset with dimensions:
  Dim{:school} Categorical{String} String[Choate, Deerfield, …, St. Paul's, Mt. Hermon] Unordered,
  Dim{:hdi_bound} Categorical{Symbol} Symbol[:lower, :upper] ForwardOrdered
and 3 layers:
  :mu    Float64 dims: Dim{:hdi_bound} (2)
  :theta Float64 dims: Dim{:school}, Dim{:hdi_bound} (8×2)
  :tau   Float64 dims: Dim{:hdi_bound} (2)
```

We can also calculate the HDI for a subset of variables:

```jldoctest hdi_infdata
hdi(idata.posterior[(:theta,)]).theta

# output

8×2 DimArray{Float64,2} theta with dimensions:
  Dim{:school} Categorical{String} String[Choate, Deerfield, …, St. Paul's, Mt. Hermon] Unordered,
  Dim{:hdi_bound} Categorical{Symbol} Symbol[:lower, :upper] ForwardOrdered
                        :lower    :upper
  "Choate"            -4.56375  17.1324
  "Deerfield"         -4.31055  14.2535
  "Phillips Andover"  -7.76922  13.6755
  "Phillips Exeter"   -4.48955  14.6635
  "Hotchkiss"         -6.46991  11.7191
  "Lawrenceville"     -7.04111  12.2087
  "St. Paul's"        -3.09262  16.2685
  "Mt. Hermon"        -5.85834  16.0143
```
"""
hdi(data::InferenceObjects.InferenceData; kwargs...) = hdi(data.posterior; kwargs...)
function hdi(data::InferenceObjects.Dataset; kwargs...)
    results = map(DimensionalData.data(data), DimensionalData.layerdims(data)) do var, dims
        x = _draw_chains_params_array(DimensionalData.DimArray(var, dims))
        r = hdi(x; kwargs...)
        lower, upper = map(Base.Fix2(_as_dimarray, x), r)
        return cat(lower, upper; dims=HDI_BOUND_DIM)
    end
    dims = Dimensions.combinedims(
        Dimensions.otherdims(data, InferenceObjects.DEFAULT_SAMPLE_DIMS), HDI_BOUND_DIM
    )
    return DimensionalData.rebuild(
        data;
        data=map(parent, results),
        dims,
        layerdims=map(Dimensions.dims, results),
        refdims=(),
        metadata=DimensionalData.NoMetadata(),
    )
end
