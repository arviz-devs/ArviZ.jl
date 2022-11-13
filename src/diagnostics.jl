@forwardfun bfmi
@forwardfun ess false
@forwardfun mcse false
@forwardfun rhat

function convert_arguments(::typeof(bfmi), data, args...; kwargs...)
    dataset = convert_to_dataset(data; group=:sample_stats)
    return tuple(dataset, args...), kwargs
end
function convert_arguments(::typeof(bfmi), data::AbstractArray, args...; kwargs...)
    dataset = convert_to_dataset((; energy=data); group=:sample_stats)
    return tuple(dataset, args...), kwargs
end

@doc doc"""
    ess(data; kwargs...) -> Dataset

Estimate the effective sample size (ESS).

The basic ESS (``N_{\mathit{eff}}``) diagnostic is computed by[^Vehtari2019][^BDA3_ESS]:
````math
\hat{N}_{\mathit{eff}} = \frac{MN}{\hat{\tau}}\\
\hat{\tau} = -1 + 2 \sum_{t'=0}^K \hat{P}_{t'},
````
where ``M`` is the number of chains, ``N`` the number of draws, ``\hat{\rho}_t`` is the
estimated autocorrelation at lag ``t``, and ``K`` is the largest integer for which
``\hat{P}_{K} = \hat{\rho}_{2K} + \hat{\rho}_{2K+1}`` is still positive.

The current implementation is similar to Stan's, which uses Geyer's initial monotone
sequence criterion[^StanManual_ESS].

See also: [`rhat`](@ref), [`mcse`](@ref), [`plot_ess`](@ref),  [`summarystats`](@ref)

[^Vehtari2019]: Rank-normalization, folding, and localization: An improved ``\hat{R}`` for
    assessing convergence of MCMC. 2019. [arXiv:1903.08008](https://arxiv.org/abs/1903.08008)[stat.CO]
[^StanManual_ESS]: https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html. Section 15.4.2
[^BDA3_ESS]: Gelman et al. Bayesian Data Analysis (2014). [PDF](http://www.stat.columbia.edu/~gelman/book/BDA3.pdf). Section 11.5

# Arguments

- `data::Any`: Any object that can be converted to an [`InferenceData`](@ref) object. See
    [`convert_to_inference_data`](@ref) for details.

# Keywords

- `var_names`: `Symbol` names of variables to include in the returned `Dataset`. Defaults to
    all.
- `method::Symbol=:bulk`: ESS method to use. Valid options are:
    + `:bulk`
    + `:tail` (specify `prob`)
    + `:quantile` (specify `prob`)
    + `:mean`: old ESS
    + `:sd`
    + `:median`
    + `:mad`: mean absolute deviance
    + `:z_scale`
    + `:folded`
    + `:identity`
    + `:local` (specify `prob`)
- `prob::Union{Real,NTuple{2,Real}}`: Probability for `:tail`, `:quantile`, or `:local` ESS
    `method`.

# Examples

Calculate the ESS using the default arguments:

```jldoctest
julia> using ArviZ

julia> data = load_example_data("non_centered_eight");

julia> ess(data)
Dataset with dimensions:
  Dim{:school} Categorical{String} String[Choate, Deerfield, …, St. Paul's, Mt. Hermon] Unordered
and 4 layers:
  :mu      Float64 dims:
  :theta_t Float64 dims: Dim{:school} (8)
  :theta   Float64 dims: Dim{:school} (8)
  :tau     Float64 dims:
```

Calculate the ESS using the `:tail` method, leaving the `prob` argument at its default
value:
```jldoctest
julia> ess(data; method=:tail)
Dataset with dimensions:
  Dim{:school} Categorical{String} String[Choate, Deerfield, …, St. Paul's, Mt. Hermon] Unordered
and 4 layers:
  :mu      Float64 dims:
  :theta_t Float64 dims: Dim{:school} (8)
  :theta   Float64 dims: Dim{:school} (8)
  :tau     Float64 dims:
```
"""
ess

"""
    mcse(data; kwargs...) -> Dataset

Calculate the Markov Chain Standard Error statistic.

See also: [`ess`](@ref), [`plot_mcse`](@ref), [`summarystats`](@ref)

# Arguments

  - `data::Any`: Any object that can be converted to an [`InferenceData`](@ref) object. See
    [`convert_to_inference_data`](@ref) for details.

# Keywords

  - `var_names`: `Symbol` names of variables to include in the returned `Dataset`. Defaults to
    all.

  - `method::Symbol=:bulk`: ESS method to use. Valid options are:
    
      + `:mean`:
      + `:sd`
      + `:median`
      + `:quantile` (specify `prob`)
  - `prob::Real`: Probability for `:quantile` ESS `method`.

# Examples

Calculate the MCSE using the default arguments:

```jldoctest
julia> using ArviZ

julia> data = load_example_data("non_centered_eight");

julia> mcse(data)
Dataset with dimensions:
  Dim{:school} Categorical{String} String[Choate, Deerfield, …, St. Paul's, Mt. Hermon] Unordered
and 4 layers:
  :mu      Float64 dims:
  :theta_t Float64 dims: Dim{:school} (8)
  :theta   Float64 dims: Dim{:school} (8)
  :tau     Float64 dims:
```

Calculate the MCSE using the `:quantile` method:

```jldoctest
julia> mcse(data; method=:quantile, prob=0.7)
Dataset with dimensions:
  Dim{:school} Categorical{String} String[Choate, Deerfield, …, St. Paul's, Mt. Hermon] Unordered
and 4 layers:
  :mu      Float64 dims:
  :theta_t Float64 dims: Dim{:school} (8)
  :theta   Float64 dims: Dim{:school} (8)
  :tau     Float64 dims:
```
"""
mcse

for f in (:ess, :mcse, :rhat)
    @eval begin
        function convert_arguments(::typeof($(f)), data, args...; kwargs...)
            dataset = convert_to_dataset(data; group=:posterior)
            return tuple(dataset, args...), kwargs
        end
        function convert_arguments(::typeof($(f)), data::AbstractArray, args...; kwargs...)
            return tuple(data, args...), kwargs
        end
        convert_result(::typeof($(f)), o::PyObject) = convert(Dataset, o)
    end
end
