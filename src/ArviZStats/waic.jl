function waic(
    data::Union{InferenceObjects.InferenceData,InferenceObjects.Dataset};
    var_name::Union{Symbol,Nothing}=nothing,
    scale::Symbol=:log,
)
    scale ∈ keys(INFORMATION_CRITERION_SCALES) || throw(
        ArgumentError("Valid `scale` values are $(keys(INFORMATION_CRITERION_SCALES))")
    )
    ll_orig = _get_log_likelihood(data; var_name)
    log_like = _draw_chains_params_array(ll_orig)
    return _waic(log_like, scale)
end

function _waic(log_like, scale)
    scale_value = INFORMATION_CRITERION_SCALES[scale]
    sample_dims = Dimensions.dims(log_like, InferenceObjects.DEFAULT_SAMPLE_DIMS)
    lpd_i = _lpd_pointwise(log_like, sample_dims)
    p_i = dropdims(
        Statistics.var(log_like; corrected=true, dims=sample_dims); dims=sample_dims
    )
    elpd_i = lpd_i - p_i
    elpd, elpd_se = _sum_and_se(elpd_i)
    p, p_se = _sum_and_se(lpd_i - elpd_i)
    ic_i = scale_value * elpd_i
    ic = scale_value * elpd
    ic_se = abs(scale_value) * elpd_se

    pointwise = InferenceObjects.Dataset((elpd=elpd_i, lpd=lpd_i, p=p_i, ic=ic_i))
    estimates = (; elpd, elpd_mcse=elpd_se, ic, ic_mcse=ic_se, p, p_mcse=p_se)
    return (; estimates, pointwise)
end
