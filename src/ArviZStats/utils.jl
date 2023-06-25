function _get_log_likelihood(
    data::InferenceObjects.InferenceData; var_name::Union{Symbol,Nothing}=nothing, kwargs...
)
    if haskey(data, :log_likelihood)
        return _get_log_likelihood(data.log_likelihood; kwargs...)
    else
        # for old InferenceData versions, log-likelihood was stored in sample_stats
        _var_name = var_name === nothing ? :log_likelihood : var_name
        return _get_log_likelihood(data.sample_stats; var_name=_var_name, kwargs...)
    end
end
function _get_log_likelihood(
    log_likelihood::InferenceObjects.Dataset; var_name::Union{Symbol,Nothing}=nothing
)
    var_name === nothing || return log_likelihood[var_name]
    var_names = keys(log_likelihood)
    length(var_names) == 1 || throw(
        ArgumentError(
            "`var_name` must be specified if there are multiple variables in `log_likelihood`",
        ),
    )
    return log_likelihood[first(var_names)]
end

function _draw_chains_params_array(x::DimensionalData.AbstractDimArray)
    sample_dims = Dimensions.dims(x, InferenceObjects.DEFAULT_SAMPLE_DIMS)
    param_dims = Dimensions.otherdims(x, sample_dims)
    dims_combined = Dimensions.combinedims(sample_dims, param_dims)
    Dimensions.dimsmatch(Dimensions.dims(x), dims_combined) && return x
    return PermutedDimsArray(x, dims_combined)
end

# compute sum and estimate of standard error of sum
function _sum_and_se(x; dims=:)
    s = sum(x; dims)
    n = dims isa Colon ? length(x) : prod(Base.Fix1(size, x), dims)
    se = StatsBase.std(x; dims) * sqrt(oftype(one(eltype(s)), n))
    return s, se
end

function _log_mean(logx, log_weights; dims=:)
    log_expectand = logx .+ log_weights
    return LogExpFunctions.logsumexp(log_expectand; dims)
end

function _se_log_mean(logx, log_weights; dims=:, log_mean=log_mean(logx, log_weights; dims))
    # variance of mean estimated using self-normalized importance weighting
    # Art B. Owen. (2013) Monte Carlo theory, methods and examples. eq. 9.9
    log_expectand = @. 2 * (log_weights + _logabssubexp(logx, log_mean))
    log_var_mean = LogExpFunctions.logsumexp(log_expectand; dims)
    # use delta method to asymptotically map variance of mean to variance of logarithm of mean
    se_log_mean = @. exp(log_var_mean / 2 - log_mean)
    return se_log_mean
end

_logabssubexp(x, y) = LogExpFunctions.logsubexp(reverse(minmax(x, y))...)

"""
    _sigdigits_matching_error(x, se; sigdigits_max=7, scale=2) -> Int

Get number of significant digits of `x` so that the last digit of `x` is the first digit of
`se*scale`.
"""
function _sigdigits_matching_error(x::Real, se::Real; sigdigits_max::Int=7, scale::Real=2)
    iszero(x) && return 0
    iszero(se) && return sigdigits_max
    first_digit_x = floor(Int, log10(abs(x)))
    last_digit_x = floor(Int, log10(se * scale))
    sigdigits_x = first_digit_x - last_digit_x + 1
    return clamp(sigdigits_x, 0, sigdigits_max)
end
