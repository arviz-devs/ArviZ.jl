"""
    loo_pit(y, y_pred, log_weights; [is_discrete,] kwargs...) -> AbstractArray

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
  - `kwargs`: Remaining keywords are forwarded to [`smooth_data`](@ref) if data is discrete.

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
"""
function loo_pit(
    y::Union{AbstractArray,Number},
    y_pred::AbstractArray,
    log_weights::AbstractArray;
    is_discrete::Union{Bool,Nothing}=nothing,
    kwargs...,
)
    sample_dims = (1, 2)
    size(log_weights) == size(y_pred) ||
        throw(ArgumentError("`log_weights` and `y_pred` must have same size"))
    _is_discrete = if is_discrete === nothing
        all(isinteger, y) && all(isinteger, y_pred)
    else
        is_discrete
    end
    if _is_discrete
        y_smooth = smooth_data(y; kwargs...)
        y_pred_smooth = smooth_data(y_pred; dims=_otherdims(y_pred, sample_dims), kwargs...)
        return _loo_pit(y_smooth, y_pred_smooth, log_weights)
    else
        return _loo_pit(y, y_pred, log_weights)
    end
end

function _loo_pit(y, y_pred, log_weights)
    sample_dims = (1, 2)
    param_dims = _otherdims(log_weights, sample_dims)
    # TODO: raise error message if sizes incompatible
    T = typeof(exp(zero(float(eltype(log_weights)))))
    if isempty(param_dims)
        return exp.(LogExpFunctions.logsumexp(log_weights[y_pred .≤ y]))
    else
        pitvals = similar(y, T)
        map!(
            pitvals,
            y,
            eachslice(y_pred; dims=param_dims),
            eachslice(log_weights; dims=param_dims),
        ) do yi, yi_hat, lw
            sel_iter = (lw_j for (lw_j, yi_hat_j) in zip(lw, yi_hat) if yi_hat_j ≤ yi)
            return exp(LogExpFunctions.logsumexp(sel_iter))
        end
    end
    return pitvals
end

# convenience methods converting from InferenceData

function _get_observed_data_key(idata::InferenceObjects.InferenceData)
    haskey(idata, :observed_data) || throw(ArgumentError("No `observed_data` group"))
    return _get_observed_data_key(idata.observed_data)
end
_get_observed_data_key(observed_data::InferenceObjects.Dataset) = only(keys(observed_data))

"""
    loo_pit(idata::InferenceData; kwargs...)

Compute LOO-PIT from groups in `idata` using PSIS-LOO.

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

See also: [`loo`](@ref), [`psis`](@ref)
"""
function loo_pit(
    idata::InferenceObjects.InferenceData;
    y_name::Union{Symbol,Nothing}=nothing,
    log_likelihood_name::Union{Symbol,Nothing}=(
        haskey(idata, :log_likelihood) ? y_name : nothing
    ),
    reff=nothing,
    kwargs...,
)
    _y_name = y_name === nothing ? _get_observed_data_key(idata) : y_name
    log_like = _draw_chains_params_array(log_likelihood(idata, log_likelihood_name))
    psis_result = _psis_loo_setup(log_like, reff)
    return loo_pit(idata, psis_result; y_name=_y_name, kwargs...)
end

"""
    loo_pit(idata::InferenceData, log_weights; kwargs...)
    loo_pit(idata::InferenceData, psis_result::PSISResult; kwargs...)
    loo_pit(idata::InferenceData, loo_result::PSISLOOResult; kwargs...)

Compute LOO-PIT values using existing normalized log LOO importance weights.

# Keywords

  - `y_name`: Name of observed data variable in `idata.observed_data`. If not provided, then
    the only observed data variable is used.
  - `y_pred_name`: Name of posterior predictive variable in `idata.posterior_predictive`.
    If not provided, then `y_name` is used.
  - `kwargs`: Remaining keywords are forwarded to [`loo_pit`](@ref).
"""
function loo_pit(
    idata::InferenceObjects.InferenceData,
    log_weights::AbstractArray;
    y_name::Union{Symbol,Nothing}=nothing,
    y_pred_name::Union{Symbol,Nothing}=nothing,
    kwargs...,
)
    haskey(idata, :observed_data) || throw(ArgumentError("No `observed_data` group"))
    haskey(idata, :posterior_predictive) ||
        throw(ArgumentError("No `posterior_predictive` group"))
    _y_name = y_name === nothing ? _get_observed_data_key(idata) : y_name
    _y_pred_name = y_pred_name === nothing ? _y_name : y_pred_name
    y = idata.observed_data[_y_name]
    y_pred = _draw_chains_params_array(idata.posterior_predictive[_y_pred_name])
    pitvals = loo_pit(y, y_pred, log_weights; kwargs...)
    return DimensionalData.rebuild(pitvals; name=Symbol("loo_pit_$(_y_name)"))
end
function loo_pit(
    idata::InferenceObjects.InferenceData, psis_result::PSIS.PSISResult; kwargs...
)
    return loo_pit(idata, _draw_chains_params_array(psis_result.log_weights); kwargs...)
end
function loo_pit(
    idata::InferenceObjects.InferenceData, loo_result::PSISLOOResult; kwargs...
)
    return loo_pit(idata, loo_result.psis_result; kwargs...)
end
