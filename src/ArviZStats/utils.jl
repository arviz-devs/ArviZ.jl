"""
    log_likelihood(data::InferenceData[, var_name]) -> DimArray
    log_likelihood(data::Dataset[, var_name]) -> DimArray

Get the log-likelihood array for the specified variable in `data`.

`var_name` must be provided if the `log_likelihood` group has more than one variable.

To support older InferenceData versions, if the `log_likelihood` group is not present, then
the `sample_stats` group is checked for a `log_likelihood` variable or for `var_name` if
provided
"""
function log_likelihood(
    data::InferenceObjects.InferenceData, var_name::Union{Symbol,Nothing}=nothing
)
    if haskey(data, :log_likelihood)
        return log_likelihood(data.log_likelihood, var_name)
    elseif haskey(data, :sample_stats)
        # for old InferenceData versions, log-likelihood was stored in sample_stats
        _var_name = var_name === nothing ? :log_likelihood : var_name
        return log_likelihood(data.sample_stats, _var_name)
    else
        throw(ArgumentError("Data must contain `log_likelihood` or `sample_stats` group"))
    end
end
function log_likelihood(
    log_like::InferenceObjects.Dataset, var_name::Union{Symbol,Nothing}=nothing
)
    if !(var_name === nothing)
        haskey(log_like, var_name) ||
            throw(ArgumentError("Variable `$(var_name)` not found in group"))
        return log_like[var_name]
    else
        var_names = keys(log_like)
        length(var_names) == 1 || throw(
            ArgumentError(
                "`var_name` must be specified if there are multiple variables in group"
            ),
        )
        return log_like[first(var_names)]
    end
end

"""
    sigdigits_matching_error(x, se; sigdigits_max=7, scale=2) -> Int

Get number of significant digits of `x` so that the last digit of `x` is the first digit of
`se*scale`.
"""
function sigdigits_matching_error(x::Real, se::Real; sigdigits_max::Int=7, scale::Real=2)
    (iszero(x) || !isfinite(x) || !isfinite(se) || !isfinite(scale)) && return 0
    sigdigits_max ≥ 0 || throw(ArgumentError("`sigdigits_max` must be non-negative"))
    se ≥ 0 || throw(ArgumentError("`se` must be non-negative"))
    iszero(se) && return sigdigits_max
    scale > 0 || throw(ArgumentError("`scale` must be positive"))
    first_digit_x = floor(Int, log10(abs(x)))
    last_digit_x = floor(Int, log10(se * scale))
    sigdigits_x = first_digit_x - last_digit_x + 1
    return clamp(sigdigits_x, 0, sigdigits_max)
end

function _draw_chains_params_array(x::DimensionalData.AbstractDimArray)
    sample_dims = Dimensions.dims(x, InferenceObjects.DEFAULT_SAMPLE_DIMS)
    param_dims = Dimensions.otherdims(x, sample_dims)
    dims_combined = Dimensions.combinedims(sample_dims, param_dims)
    Dimensions.dimsmatch(Dimensions.dims(x), dims_combined) && return x
    return PermutedDimsArray(x, dims_combined)
end

_logabssubexp(x, y) = LogExpFunctions.logsubexp(reverse(minmax(x, y))...)

# compute sum and estimate of standard error of sum
function _sum_and_se(x; dims=:)
    s = sum(x; dims)
    n = dims isa Colon ? length(x) : prod(Base.Fix1(size, x), dims)
    se = StatsBase.std(x; dims) * sqrt(oftype(one(eltype(s)), n))
    return s, se
end
_sum_and_se(x::Number; kwargs...) = (x, oftype(float(x), NaN))

function _log_mean(logx, log_weights; dims=:)
    log_expectand = logx .+ log_weights
    return LogExpFunctions.logsumexp(log_expectand; dims)
end

function _se_log_mean(
    logx, log_weights; dims=:, log_mean=_log_mean(logx, log_weights; dims)
)
    # variance of mean estimated using self-normalized importance weighting
    # Art B. Owen. (2013) Monte Carlo theory, methods and examples. eq. 9.9
    log_expectand = @. 2 * (log_weights + _logabssubexp(logx, log_mean))
    log_var_mean = LogExpFunctions.logsumexp(log_expectand; dims)
    # use delta method to asymptotically map variance of mean to variance of logarithm of mean
    se_log_mean = @. exp(log_var_mean / 2 - log_mean)
    return se_log_mean
end
