@forwardfun compare
@forwardfun hdi
@forwardfun loo
@forwardfun loo_pit
@forwardfun r2_score
@forwardfun waic

for f in (:loo, :waic)
    @eval begin
        function convert_arguments(::typeof($(f)), data, args...; kwargs...)
            idata = convert_to_inference_data(data)
            return tuple(idata, args...), kwargs
        end
    end
end
function convert_arguments(::typeof(compare), data, args...; kwargs...)
    dict = Dict(k => try
        topandas(Val(:ELPDData), v)
    catch
        convert_to_inference_data(v)
    end for (k, v) in pairs(data))
    return tuple(dict, args...), kwargs
end

convert_result(::typeof(loo), result) = todataframes(result)
convert_result(::typeof(waic), result) = todataframes(result)
convert_result(::typeof(r2_score), result) = todataframes(result)
function convert_result(::typeof(compare), result)
    return todataframes(result; index_name=:name)
end

"""
    psislw(log_weights, reff=1.0) -> (lw_out, kss)

Pareto smoothed importance sampling (PSIS).

!!! note
    
    This function is deprecated and is just a thin wrapper around [`psis`](@ref).

# Arguments

  - `log_weights`: Array of size `(nobs, ndraws)`
  - `reff`: relative MCMC efficiency, `ess / n`

# Returns

  - `lw_out`: Smoothed log weights
  - `kss`: Pareto tail indices
"""
function psislw(logw, reff=1)
    @warn "`psislw(logw[, reff])` is deprecated, use `psis(logw[, reff])`" maxlog = 1
    result = psis(logw, reff)
    log_weights = result.log_weights
    d = ndims(log_weights)
    dims = d == 1 ? Colon() : ntuple(Base.Fix1(+, 1), d - 1)
    log_norm_exp = logsumexp(log_weights; dims)
    log_weights .-= log_norm_exp
    return log_weights, result.pareto_shape
end

@doc doc"""
    summarystats(
        data::InferenceData;
        group = :posterior,
        kwargs...,
    ) -> Union{Dataset,DataFrames.DataFrame}
    summarystats(data::Dataset; kwargs...) -> Union{Dataset,DataFrames.DataFrame}

Compute summary statistics on `data`.

# Arguments

- `data::Union{Dataset,InferenceData}`: The data on which to compute summary statistics. If
    `data` is an [`InferenceData`](@ref), only the dataset corresponding to `group` is used.

# Keywords

- `var_names`: Collection of names of variables as `Symbol`s to include in summary
- `include_circ::Bool=false`: Whether to include circular statistics
- `digits::Int`: Number of decimals used to round results. If not provided, numbers are not
    rounded.
- `stat_funcs::Union{Dict{String,Function},Vector{Function}}=nothing`: A vector of functions
    or a dict of functions with function names as keys used to calculate statistics. By
    default, the mean, standard deviation, simulation standard error, and highest posterior
    density intervals are included.
    The functions will be given one argument, the samples for a variable as an array, The
    functions should operate on an array, returning a single number. For example,
    `Statistics.mean`, or `Statistics.var` would both work.
- `extend::Bool=true`: If `true`, use the statistics returned by `stat_funcs` in addition
    to, rather than in place of, the default statistics. This is only meaningful when
    `stat_funcs` is not `nothing`.
- `hdi_prob::Real=0.94`: HDI interval to compute. This is only meaningful when `stat_funcs`
    is `nothing`.
- `skipna::Bool=false`: If `true`, ignores `NaN` values when computing the summary
    statistics. It does not affect the behaviour of the functions passed to `stat_funcs`.

# Returns

- `DataFrames.DataFrame`: Summary statistics for each variable. Default statistics are:
    + `mean`
    + `sd`
    + `hdi_3%`
    + `hdi_97%`
    + `mcse_mean`
    + `mcse_sd`
    + `ess_bulk`
    + `ess_tail`
    + `r_hat` (only computed for traces with 2 or more chains)

# Examples

```@example summarystats
using ArviZ
idata = load_example_data("centered_eight")
summarystats(idata; var_names=(:mu, :tau))
```

Other statistics can be calculated by passing a list of functions or a dictionary with key,
function pairs:

```@example summarystats
using Statistics
function median_sd(x)
    med = median(x)
    sd = sqrt(mean((x .- med).^2))
    return sd
end

func_dict = Dict(
    "std" => x -> std(x; corrected = false),
    "median_std" => median_sd,
    "5%" => x -> quantile(x, 0.05),
    "median" => median,
    "95%" => x -> quantile(x, 0.95),
)

summarystats(idata; var_names = (:mu, :tau), stat_funcs = func_dict, extend = false)
```
"""
function StatsBase.summarystats(data::InferenceData; group::Symbol=:posterior, kwargs...)
    dataset = getproperty(data, group)
    return summarystats(dataset; kwargs...)
end
function StatsBase.summarystats(
    data::Dataset; var_names=nothing, digits::Int=typemax(Int), kwargs...
)
    var_names = var_names === nothing ? var_names : collect(var_names)
    round_to = digits == typemax(Int) ? nothing : digits
    s = arviz.summary(data; var_names, round_to, kwargs...)
    return todataframes(s; index_name=:variable)
end

"""
    summary(
        data; group = :posterior, coords dims, kwargs...,
    ) -> Union{Dataset,DataFrames.DataFrame}

Compute summary statistics on any object that can be passed to [`convert_to_dataset`](@ref).

# Keywords

  - `coords`: Map from named dimension to named indices.
  - `dims`: Map from variable name to names of its dimensions.
  - `kwargs`: Keyword arguments passed to [`summarystats`](@ref).
"""
function summary(data; group=:posterior, coords=(;), dims=(;), kwargs...)
    dataset = convert_to_dataset(data; group, coords, dims)
    return summarystats(dataset; kwargs...)
end
