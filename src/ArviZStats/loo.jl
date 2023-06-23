function loo(
    data::Union{InferenceObjects.InferenceData,InferenceObjects.Dataset};
    var_name::Union{Symbol,Nothing}=nothing,
    reff=nothing,
    scale::Symbol=:log,
    kwargs...,
)
    scale ∈ keys(INFORMATION_CRITERION_SCALES) || throw(
        ArgumentError("Valid `scale` values are $(keys(INFORMATION_CRITERION_SCALES))")
    )
    log_like, reff_new, psis_result = _psis_loo_setup(data, var_name, reff; kwargs...)
    return _loo(log_like, reff_new, scale, psis_result)
end

function _psis_loo_setup(data, var_name, _reff; kwargs...)
    ll_orig = _get_log_likelihood(data; var_name)
    log_like = _draw_chains_params_array(ll_orig)
    if _reff === nothing
        # normalize log likelihoods to improve numerical stability of ESS estimate
        like = LogExpFunctions.softmax(log_like; dims=InferenceObjects.DEFAULT_SAMPLE_DIMS)
        reff = ess(like; kind=:basic, split_chains=1, relative=true)
    else
        reff = _reff
    end
    # smooth importance weights
    psis_result = PSIS.psis(-log_like, reff; kwargs...)
    return log_like, reff, psis_result
end

function _loo(log_like, reff, scale; kwargs...)
    scale_value = INFORMATION_CRITERION_SCALES[scale]
    sample_dims = Dimensions.dims(log_like, InferenceObjects.DEFAULT_SAMPLE_DIMS)
    psis_result = PSIS.psis(-log_like, reff; kwargs...)
    elpd_i, elpd_se_i = _elpd_loo_pointwise_and_se(psis_result, log_like, sample_dims)
    elpd, elpd_se = _sum_and_se(elpd_i)
    lpd_i = _lpd_pointwise(log_like, sample_dims)
    p_i = lpd_i .- elpd_i ./ scale_value
    p, p_se = _sum_and_se(p_i)
    ic_i = scale_value * elpd_i
    ic = scale_value * elpd
    ic_se = abs(scale_value) * elpd_se
    pointwise = InferenceObjects.Dataset((
        elpd=elpd_i,
        elpd_mcse=elpd_se_i,
        lpd=lpd_i,
        p=p_i,
        ic=ic_i,
        reff,
        pareto_shape=psis_result.pareto_shape,
    ))
    estimates = (; elpd, elpd_mcse=elpd_se, ic, ic_mcse=ic_se, p, p_mcse=p_se)
    return (; estimates, pointwise, psis_result)
end

function _lpd_pointwise(log_likelihood, dims)
    ndraws = prod(Base.Fix1(size, log_likelihood), dims)
    lpd = LogExpFunctions.logsumexp(log_likelihood; dims)
    T = eltype(lpd)
    return dropdims(lpd; dims) .- log(T(ndraws))
end

function _elpd_loo_pointwise_and_se(psis_result::PSIS.PSISResult, log_likelihood, dims)
    log_norm = LogExpFunctions.logsumexp(psis_result.log_weights; dims)
    log_weights = (psis_result.log_weights .- log_norm) .+ log_likelihood
    # NOTE: loo takes a different approach to estimate mcses for elpd_i, which gives a very
    # different result. TODO: work out why this is the case.
    elpd_i, elpd_i_se = _logsumexp_and_se(log_weights; dims)
    return (dropdims(elpd_i; dims), dropdims(elpd_i_se; dims) ./ sqrt.(psis_result.reff))
end
