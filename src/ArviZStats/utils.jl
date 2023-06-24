function _get_log_likelihood(data::InferenceObjects.InferenceData; kwargs...)
    return _get_log_likelihood(data.log_likelihood; kwargs...)
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

# compute lpogsumexp and estimate of standard error of lpogsumexp
function _logsumexp_and_se(x; dims=:)
    # use the delta method to approximate the standard error of logsumexp
    # i.e. if z ~ N(μ, σ²/n), then log(z) ~ N(log(μ), (σ / μ)² / n)
    logs = LogExpFunctions.logsumexp(x; dims)
    logs2 = LogExpFunctions.logsumexp(2x; dims)
    n = dims isa Colon ? length(x) : prod(Base.Fix1(size, x), dims)
    invn = 1//n
    logs_se = @. sqrt(exp(logs2 - 2logs) - invn)
    return logs, logs_se
end
