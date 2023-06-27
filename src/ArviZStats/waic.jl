"""
$(SIGNATURES)

Results of computing the widely applicable information criterion (WAIC).

See also: [`waic`](@ref), [`ELPDResult`](@ref)

$(FIELDS)
"""
struct WAICResult{E,P} <: AbstractELPDResult
    "(E)LPD estimates"
    estimates::E
    "Pointwise (E)LPD estimates"
    pointwise::P
end

function elpd_estimates(r::WAICResult; pointwise::Bool=false)
    return pointwise ? r.pointwise : r.estimates
end

function Base.show(io::IO, mime::MIME"text/plain", result::WAICResult)
    println(io, "WAICResult with estimates")
    _print_elpd_estimates(io, mime, result)
    return nothing
end

"""
    waic(data; var_name=nothing) -> WAICResult

Compute the widely applicable information criterion (WAIC).[^Watanabe2010][^Vehtari2017][^LOOFAQ]

`data` is either an [`InferenceData`](@ref) or a [`Dataset`](@ref) containing log-likelihood
values.

# Keywords

  - `var_name::Union{Nothing,Symbol}`: Name of the variable in `data` containing the
    log-likelihood values. This must be provided if more than one variable is present.

# Returns

  - [`WAICResult`](@ref)

[^Watanabe2010]: Watanabe, S. Asymptotic Equivalence of Bayes Cross Validation and Widely Applicable Information Criterion in Singular Learning Theory. 11(116):3571−3594, 2010. https://jmlr.csail.mit.edu/papers/v11/watanabe10a.html
[^Vehtari2017]: Vehtari, A., Gelman, A. & Gabry, J.
    Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC.
    Stat Comput 27, 1413–1432 (2017).
    doi: [10.1007/s11222-016-9696-4](https://doi.org/10.1007/s11222-016-9696-4)
    arXiv: [1507.04544](https://arxiv.org/abs/1507.04544)
[^LOOFAQ]: Aki Vehtari. Cross-validation FAQ. https://mc-stan.org/loo/articles/online-only/faq.html
"""
function waic(
    data::Union{InferenceObjects.InferenceData,InferenceObjects.Dataset};
    var_name::Union{Symbol,Nothing}=nothing,
)
    ll_orig = log_likelihood(data, var_name)
    log_like = _draw_chains_params_array(ll_orig)
    return _waic(log_like)
end

function _waic(log_like)
    sample_dims = Dimensions.dims(log_like, InferenceObjects.DEFAULT_SAMPLE_DIMS)

    # compute pointwise estimates
    lpd_i = _lpd_pointwise(log_like, sample_dims)
    p_i = dropdims(
        Statistics.var(log_like; corrected=true, dims=sample_dims); dims=sample_dims
    )
    elpd_i = lpd_i - p_i
    pointwise = InferenceObjects.Dataset((elpd=elpd_i, lpd=lpd_i))

    # combine estimates
    estimates = _elpd_estimates_from_pointwise(pointwise)

    return WAICResult(estimates, pointwise)
end
