"""
$(SIGNATURES)

Results of Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO).

See also: [`loo`](@ref), [`ELPDResult`](@ref)

$(FIELDS)
"""
struct PSISLOOResult{E,P,R<:PSIS.PSISResult} <: AbstractELPDResult
    "(E)LPD estimates"
    estimates::E
    "Pointwise (E)LPD estimates"
    pointwise::P
    "Pareto-smoothed importance sampling results"
    psis_result::R
end

function elpd_estimates(r::PSISLOOResult; pointwise::Bool=false)
    return pointwise ? r.pointwise : r.estimates
end

function Base.show(io::IO, mime::MIME"text/plain", result::PSISLOOResult)
    println(io, "PSISLOOResult with estimates")
    _print_elpd_estimates(io, mime, result)
    println(io)
    print(io, "and ")
    show(io, mime, result.psis_result)
    return nothing
end

"""
    loo(data; var_name=nothing, reff=nothing, kwargs...) -> PSISLOOResult

Compute the Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO).
[^Vehtari2017][^LOOFAQ]

`data` is either an [`InferenceData`](@ref) or a [`Dataset`](@ref) containing log-likelihood
values.

# Keywords

  - `var_name::Union{Nothing,Symbol}`: Name of the variable in `data` containing the
    log-likelihood values. This must be provided if more than one variable is present.
  - `reff::Union{Real,AbstractArray{<:Real}}`: The relative effective sample size(s) of the
    _likelihood_ values. If an array, it must have the same data dimensions as the
    corresponding log-likelihood variable. If not provided, then this is estimated using
    [`ess`](@ref).
  - `kwargs`: Remaining keywords are forwarded to [`psis`](@ref).

# Returns

  - [`PSISLOOResult`](@ref)

[^Vehtari2017]: Vehtari, A., Gelman, A. & Gabry, J.
    Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC.
    Stat Comput 27, 1413–1432 (2017).
    doi: [10.1007/s11222-016-9696-4](https://doi.org/10.1007/s11222-016-9696-4)
    arXiv: [1507.04544](https://arxiv.org/abs/1507.04544)
[^LOOFAQ]: Aki Vehtari. Cross-validation FAQ. https://mc-stan.org/loo/articles/online-only/faq.html
"""
function loo(
    data::Union{InferenceObjects.InferenceData,InferenceObjects.Dataset};
    var_name::Union{Symbol,Nothing}=nothing,
    reff=nothing,
    kwargs...,
)
    log_like, reff_new, psis_result = _psis_loo_setup(data, var_name, reff; kwargs...)
    return _loo(log_like, reff_new, psis_result)
end

function _psis_loo_setup(data, var_name, _reff; kwargs...)
    ll_orig = get_log_likelihood(data; var_name)
    log_like = _draw_chains_params_array(ll_orig)
    if _reff === nothing
        # normalize log likelihoods to improve numerical stability of ESS estimate
        like = LogExpFunctions.softmax(log_like; dims=InferenceObjects.DEFAULT_SAMPLE_DIMS)
        reff = MCMCDiagnosticTools.ess(like; kind=:basic, split_chains=1, relative=true)
    else
        reff = _reff
    end
    # smooth importance weights
    psis_result = PSIS.psis(-log_like, reff; kwargs...)
    return log_like, reff, psis_result
end

function _loo(log_like, reff, psis_result)
    sample_dims = Dimensions.dims(log_like, InferenceObjects.DEFAULT_SAMPLE_DIMS)

    # compute pointwise estimates
    lpd_i = _lpd_pointwise(log_like, sample_dims)
    elpd_i, elpd_se_i = _elpd_loo_pointwise_and_se(psis_result, log_like, sample_dims)
    pointwise = InferenceObjects.convert_to_dataset((
        elpd=elpd_i,
        elpd_mcse=elpd_se_i,
        lpd=lpd_i,
        reff,
        pareto_shape=psis_result.pareto_shape,
    ))

    # combine estimates
    estimates = _elpd_estimates_from_pointwise(pointwise)

    return PSISLOOResult(estimates, pointwise, psis_result)
end

function _elpd_loo_pointwise_and_se(psis_result::PSIS.PSISResult, log_likelihood, dims)
    log_norm = LogExpFunctions.logsumexp(psis_result.log_weights; dims)
    log_weights = psis_result.log_weights .- log_norm
    elpd_i = _log_mean(log_likelihood, log_weights; dims)
    elpd_i_se = _se_log_mean(log_likelihood, log_weights; dims, log_mean=elpd_i)
    return (
        elpd=dropdims(elpd_i; dims),
        elpd_se=dropdims(elpd_i_se; dims) ./ sqrt.(psis_result.reff),
    )
end
