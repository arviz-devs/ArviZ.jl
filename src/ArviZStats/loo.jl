"""
$(SIGNATURES)

Results of Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO).

See also: [`loo`](@ref), [`ELPDResult`](@ref)

$(FIELDS)
"""
struct PSISLOOResult{E,P,R<:PSIS.PSISResult} <: AbstractELPDResult
    "Estimates"
    estimates::E
    "Pointwise estimates"
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
    loo(log_likelihood; reff=nothing, kwargs...) -> PSISLOOResult

Compute the Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO).
[^Vehtari2017][^LOOFAQ]

`log_likelihood` must be an array of log-likelihood values with shape
`(chains, draws[, params...])`.

# Keywords

  - `reff::Union{Real,AbstractArray{<:Real}}`: The relative effective sample size(s) of the
    _likelihood_ values. If an array, it must have the same data dimensions as the
    corresponding log-likelihood variable. If not provided, then this is estimated using
    [`ess`](@ref).
  - `kwargs`: Remaining keywords are forwarded to [`psis`](@ref).

See also: [`PSISLOOResult`](@ref), [`waic`](@ref)

[^Vehtari2017]: Vehtari, A., Gelman, A. & Gabry, J.
    Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC.
    Stat Comput 27, 1413–1432 (2017).
    doi: [10.1007/s11222-016-9696-4](https://doi.org/10.1007/s11222-016-9696-4)
    arXiv: [1507.04544](https://arxiv.org/abs/1507.04544)
[^LOOFAQ]: Aki Vehtari. Cross-validation FAQ. https://mc-stan.org/loo/articles/online-only/faq.html
"""
loo(ll::AbstractArray; kwargs...) = _loo(ll; kwargs...)

"""
    loo(data::Dataset; [var_name::Symbol,] kwargs...) -> PSISLOOResult
    loo(data::InferenceData; [var_name::Symbol,] kwargs...) -> PSISLOOResult

Compute PSIS-LOO from log-likelihood values in `data`.

If more than one log-likelihood variable is present, then `var_name` must be provided.
"""
function loo(
    data::Union{InferenceObjects.InferenceData,InferenceObjects.Dataset};
    var_name::Union{Symbol,Nothing}=nothing,
    kwargs...,
)
    log_like = _draw_chains_params_array(log_likelihood(data, var_name))
    result = loo(log_like; kwargs...)
    pointwise = ArviZ.convert_to_dataset(result.pointwise)
    return PSISLOOResult(result.estimates, pointwise, result.psis_result)
end

function _psis_loo_setup(log_like, _reff; kwargs...)
    if _reff === nothing
        # normalize log likelihoods to improve numerical stability of ESS estimate
        like = LogExpFunctions.softmax(log_like; dims=(1, 2))
        reff = MCMCDiagnosticTools.ess(like; kind=:basic, split_chains=1, relative=true)
    else
        reff = _reff
    end
    # smooth importance weights
    psis_result = PSIS.psis(-log_like, reff; kwargs...)
    return psis_result
end

function _loo(log_like; reff=nothing, kwargs...)
    psis_result = _psis_loo_setup(log_like, reff; kwargs...)
    return _loo(log_like, psis_result)
end
function _loo(log_like, psis_result, dims=(1, 2))
    # compute pointwise estimates
    lpd_i = _lpd_pointwise(log_like, dims)
    elpd_i, elpd_se_i = _elpd_loo_pointwise_and_se(psis_result, log_like, dims)
    p_i = lpd_i - elpd_i
    pointwise = (;
        elpd=elpd_i,
        elpd_mcse=elpd_se_i,
        p=p_i,
        reff=psis_result.reff,
        pareto_shape=psis_result.pareto_shape,
    )

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
        elpd=_maybe_scalar(dropdims(elpd_i; dims)),
        elpd_se=_maybe_scalar(dropdims(elpd_i_se; dims) ./ sqrt.(psis_result.reff)),
    )
end

function ArviZ.topandas(::Val{:ELPDData}, d::PSISLOOResult)
    estimates = elpd_estimates(d)
    pointwise = elpd_estimates(d; pointwise=true)
    psis_result = d.psis_result
    n_samples = psis_result.nchains * psis_result.ndraws
    n_data_points = psis_result.nparams
    warn_mg = ""
    ds = ArviZ.convert_to_dataset((
        loo_i=pointwise.elpd, pareto_shape=pointwise.pareto_shape
    ))
    pyds = PyCall.PyObject(ds)
    return PyCall.pycall(
        ArviZ.arviz.stats.ELPDData,
        PyCall.PyObject;
        data=[
            estimates.elpd,
            estimates.elpd_mcse,
            estimates.p,
            n_samples,
            n_data_points,
            warn_mg,
            pyds.loo_i,
            pyds.pareto_shape,
            "log",
        ],
        index=[
            "elpd_loo",
            "se",
            "p_loo",
            "n_samples",
            "n_data_points",
            "warning",
            "loo_i",
            "pareto_k",
            "scale",
        ],
    )
end
