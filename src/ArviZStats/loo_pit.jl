"""
    loo_pit(y, y_pred, log_weights; kwargs...) -> Union{Real,AbstractArray}

Compute leave-one-out probability integral transform (LOO-PIT) checks.

# Arguments

  - `y`: array of observations with shape `(params...,)`
  - `y_pred`: array of posterior predictive samples with shape `(draws, chains, params...)`.
  - `log_weights`: array of normalized log LOO importance weights with shape
    `(draws, chains, params...)`.

# Keywords

  - `is_discrete`: If not provided, then it is set to `true` iff elements of `y` and `y_pred`
    are all integer-valued. If `true`, then data are smoothed using [`smooth_data`](@ref) to
    make them non-discrete before estimating LOO-PIT values.
  - `kwargs`: Remaining keywords are forwarded to `smooth_data` if data is discrete.

# Returns

  - `pitvals`: LOO-PIT values with same size as `y`. If `y` is a scalar, then `pitvals` is a
    scalar.

LOO-PIT is a marginal posterior predictive check. If ``y_{-i}`` is the array ``y`` of
observations with the ``i``th observation left out, and ``y_i^*`` is a posterior prediction
of the ``i``th observation, then the LOO-PIT value for the ``i``th observation is defined as

```math
P(y_i^* \\le y_i \\mid y_{-i}) = \\int_{-\\infty}^{y_i} p(y_i^* \\mid y_{-i}) \\mathrm{d} y_i^*
```

The LOO posterior predictions and the corresponding observations should have similar
distributions, so if conditional predictive distributions are well-calibrated, then all
LOO-PIT values should be approximately uniformly distributed on ``[0, 1]``.[^Gabry2019]

[^Gabry2019]: Gabry, J., Simpson, D., Vehtari, A., Betancourt, M. & Gelman, A.
    Visualization in Bayesian Workflow.
    J. R. Stat. Soc. Ser. A Stat. Soc. 182, 389–402 (2019).
    doi: [10.1111/rssa.12378](https://doi.org/10.1111/rssa.12378)
    arXiv: [1709.01449](https://arxiv.org/abs/1709.01449)

# Examples

Calculate LOO-PIT values using as test quantity the observed values themselves.

```jldoctest loo_pit1
using ArviZ, ArviZExampleData
idata = load_example_data("centered_eight")
log_weights = loo(idata; var_name=:obs).psis_result.log_weights
loo_pit(
    idata.observed_data.obs,
    permutedims(idata.posterior_predictive.obs, (:draw, :chain, :school)),
    log_weights,
)

# output

8-element DimArray{Float64,1} with dimensions:
  Dim{:school} Categorical{String} String[Choate, Deerfield, …, St. Paul's, Mt. Hermon] Unordered
 "Choate"            0.943511
 "Deerfield"         0.63797
 "Phillips Andover"  0.316697
 "Phillips Exeter"   0.582252
 "Hotchkiss"         0.295321
 "Lawrenceville"     0.403318
 "St. Paul's"        0.902508
 "Mt. Hermon"        0.655275
```

Calculate LOO-PIT values using as test quantity the square of the difference between
each observation and `mu`.

```jldoctest loo_pit1
using DimensionalData, Statistics
T = idata.observed_data.obs .- only(median(idata.posterior.mu; dims=(:draw, :chain)))
T_pred = permutedims(
    broadcast_dims(-, idata.posterior_predictive.obs, idata.posterior.mu),
    (:draw, :chain, :school),
)
loo_pit(T .^ 2, T_pred .^ 2, log_weights)

# output

8-element DimArray{Float64,1} with dimensions:
  Dim{:school} Categorical{String} String[Choate, Deerfield, …, St. Paul's, Mt. Hermon] Unordered
 "Choate"            0.873577
 "Deerfield"         0.243686
 "Phillips Andover"  0.357563
 "Phillips Exeter"   0.149908
 "Hotchkiss"         0.435094
 "Lawrenceville"     0.220627
 "St. Paul's"        0.775086
 "Mt. Hermon"        0.296706
```
"""
function loo_pit(
    y::Union{AbstractArray,Number},
    y_pred::AbstractArray,
    log_weights::AbstractArray;
    is_discrete::Union{Bool,Nothing}=nothing,
    kwargs...,
)
    sample_dims = (1, 2)
    size(y) == size(y_pred)[3:end] ||
        throw(ArgumentError("data dimensions of `y` and `y_pred` must have the size"))
    size(log_weights) == size(y_pred) ||
        throw(ArgumentError("`log_weights` and `y_pred` must have same size"))
    _is_discrete = if is_discrete === nothing
        all(isinteger, y) && all(isinteger, y_pred)
    else
        is_discrete
    end
    if _is_discrete
        is_discrete === nothing &&
            @warn "All data and predictions are integer-valued. Smoothing data before running `loo_pit`."
        y_smooth = smooth_data(y; kwargs...)
        y_pred_smooth = smooth_data(y_pred; dims=_otherdims(y_pred, sample_dims), kwargs...)
        return _loo_pit(y_smooth, y_pred_smooth, log_weights)
    else
        return _loo_pit(y, y_pred, log_weights)
    end
end

"""
    loo_pit(idata::InferenceData, log_weights; kwargs...) -> DimArray

Compute LOO-PIT values using existing normalized log LOO importance weights.

# Keywords

  - `y_name`: Name of observed data variable in `idata.observed_data`. If not provided, then
    the only observed data variable is used.
  - `y_pred_name`: Name of posterior predictive variable in `idata.posterior_predictive`.
    If not provided, then `y_name` is used.
  - `kwargs`: Remaining keywords are forwarded to [`loo_pit`](@ref).

# Examples

Calculate LOO-PIT values using already computed log weights.

```jldoctest
using ArviZ, ArviZExampleData
idata = load_example_data("centered_eight")
loo_result = loo(idata; var_name=:obs)
loo_pit(idata, loo_result.psis_result.log_weights; y_name=:obs)

# output

8-element DimArray{Float64,1} loo_pit_obs with dimensions:
  Dim{:school} Categorical{String} String[Choate, Deerfield, …, St. Paul's, Mt. Hermon] Unordered
 "Choate"            0.943511
 "Deerfield"         0.63797
 "Phillips Andover"  0.316697
 "Phillips Exeter"   0.582252
 "Hotchkiss"         0.295321
 "Lawrenceville"     0.403318
 "St. Paul's"        0.902508
 "Mt. Hermon"        0.655275
```
"""
function loo_pit(
    idata::InferenceObjects.InferenceData,
    log_weights::AbstractArray;
    y_name::Union{Symbol,Nothing}=nothing,
    y_pred_name::Union{Symbol,Nothing}=nothing,
    kwargs...,
)
    (_y_name, y), (_, _y_pred) = observations_and_predictions(idata, y_name, y_pred_name)
    y_pred = _draw_chains_params_array(_y_pred)
    pitvals = loo_pit(y, y_pred, log_weights; kwargs...)
    return DimensionalData.rebuild(pitvals; name=Symbol("loo_pit_$(_y_name)"))
end

"""
    loo_pit(idata::InferenceData; kwargs...) -> DimArray

Compute LOO-PIT from groups in `idata` using PSIS-LOO.

See also: [`loo`](@ref), [`psis`](@ref)

# Keywords

  - `y_name`: Name of observed data variable in `idata.observed_data`. If not provided, then
    the only observed data variable is used.
  - `y_pred_name`: Name of posterior predictive variable in `idata.posterior_predictive`.
    If not provided, then `y_name` is used.
  - `log_likelihood_name`: Name of log-likelihood variable in `idata.log_likelihood`.
    If not provided, then `y_name` is used if `idata` has a `log_likelihood` group,
    otherwise the only variable is used.
  - `reff::Union{Real,AbstractArray{<:Real}}`: The relative effective sample size(s) of the
    _likelihood_ values. If an array, it must have the same data dimensions as the
    corresponding log-likelihood variable. If not provided, then this is estimated using
    [`ess`](@ref).
  - `kwargs`: Remaining keywords are forwarded to [`loo_pit`](@ref).

# Examples

Calculate LOO-PIT values using as test quantity the observed values themselves.

```jldoctest
using ArviZ, ArviZExampleData
idata = load_example_data("centered_eight")
loo_pit(idata; y_name=:obs)

# output

8-element DimArray{Float64,1} loo_pit_obs with dimensions:
  Dim{:school} Categorical{String} String[Choate, Deerfield, …, St. Paul's, Mt. Hermon] Unordered
 "Choate"            0.943511
 "Deerfield"         0.63797
 "Phillips Andover"  0.316697
 "Phillips Exeter"   0.582252
 "Hotchkiss"         0.295321
 "Lawrenceville"     0.403318
 "St. Paul's"        0.902508
 "Mt. Hermon"        0.655275
```
"""
function loo_pit(
    idata::InferenceObjects.InferenceData;
    y_name::Union{Symbol,Nothing}=nothing,
    y_pred_name::Union{Symbol,Nothing}=nothing,
    log_likelihood_name::Union{Symbol,Nothing}=nothing,
    reff=nothing,
    kwargs...,
)
    (_y_name, y), (_, _y_pred) = observations_and_predictions(idata, y_name, y_pred_name)
    y_pred = _draw_chains_params_array(_y_pred)
    if log_likelihood_name === nothing
        if haskey(idata, :log_likelihood)
            _log_like = log_likelihood(idata.log_likelihood, _y_name)
        elseif haskey(idata, :sample_stats) && haskey(idata.sample_stats, :log_likelihood)
            _log_like = idata.sample_stats.log_likelihood
        else
            throw(ArgumentError("There must be a `log_likelihood` group in `idata`"))
        end
    else
        _log_like = log_likelihood(idata.log_likelihood, log_likelihood_name)
    end
    log_like = _draw_chains_params_array(_log_like)
    psis_result = _psis_loo_setup(log_like, reff)
    pitvals = loo_pit(y, y_pred, psis_result.log_weights; kwargs...)
    return DimensionalData.rebuild(pitvals; name=Symbol("loo_pit_$(_y_name)"))
end

function _loo_pit(y::Number, y_pred, log_weights)
    return @views exp.(LogExpFunctions.logsumexp(log_weights[y_pred .≤ y]))
end
function _loo_pit(y::AbstractArray, y_pred, log_weights)
    sample_dims = (1, 2)
    T = typeof(exp(zero(float(eltype(log_weights)))))
    pitvals = similar(y, T)
    param_dims = _otherdims(log_weights, sample_dims)
    # work around for `eachslices` not supporting multiple dims in older Julia versions
    map!(
        pitvals,
        y,
        CartesianIndices(map(Base.Fix1(axes, y_pred), param_dims)),
        CartesianIndices(map(Base.Fix1(axes, log_weights), param_dims)),
    ) do yi, i1, i2
        yi_pred = @views y_pred[:, :, i1]
        lwi = @views log_weights[:, :, i2]
        init = T(-Inf)
        sel_iter = Iterators.flatten((
            init, (lwi_j for (lwi_j, yi_pred_j) in zip(lwi, yi_pred) if yi_pred_j ≤ yi)
        ))
        return clamp(exp(LogExpFunctions.logsumexp(sel_iter)), 0, 1)
    end
    return pitvals
end
