const sample_stats_types = Dict(
    "mean_tree_accept" => Float64,
    "energy" => Float64,
    "energy_error" => Float64,
    "max_energy_error" => Float64,
    "step_size" => Float64,
    "step_size_bar" => Float64,
    "tree_size" => Int,
    "depth" => Int,
    "tune" => Bool,
    "diverging" => Bool,
)

@doc forwarddoc(:compare) compare(args...; kwargs...) =
    arviz.compare(args...; kwargs...) |> Pandas.DataFrame

Docs.getdoc(::typeof(compare)) = forwardgetdoc(:compare)

@forwardfun hpd

@doc forwarddoc(:loo) loo(args...; kwargs...) =
    arviz.loo(args...; kwargs...) |> Pandas.Series

Docs.getdoc(::typeof(loo)) = forwardgetdoc(:loo)

@forwardfun loo_pit

@forwardfun psislw

@doc forwarddoc(:r2_score) r2_score(args...; kwargs...) =
    arviz.r2_score(args...; kwargs...) |> Pandas.Series

Docs.getdoc(::typeof(r2_score)) = forwardgetdoc(:r2_score)

@doc forwarddoc(:waic) waic(args...; kwargs...) =
    arviz.waic(args...; kwargs...) |> Pandas.Series

Docs.getdoc(::typeof(waic)) = forwardgetdoc(:waic)

"""
    summarystats(data::Dataset; kwargs...) -> Union{Pandas.DataFrame,Dataset}
    summarystats(data::InferenceData; group = :posterior kwargs...) -> Union{Pandas.DataFrame,Dataset}

Compute summary statistics on `data`.

# Arguments
- `data::Union{Dataset,InferenceData}`: The data on which to compute summary
      statistics. If `data` is an [`InferenceData`](@ref), only the dataset
      corresponding to `group` is used.

# Keywords
- `var_names::Vector{String}=nothing`: Names of variables to include in summary
- `include_circ::Bool=false`: Whether to include circular statistics
- `fmt::String="wide"`: Return format is either `Pandas.DataFrame` ("wide", "long")
      or [`Dataset`](@ref) ("xarray").
- `round_to::Int=nothing`: Number of decimals used to round results.
      Use `nothing` to return raw numbers.
- `stat_funcs::Union{Dict{String,Function},Vector{Function}}=nothing`:
      A vector of functions or a dict of functions with function names as keys
      used to calculate statistics. By default, the mean, standard deviation,
      simulation standard error, and highest posterior density intervals are
      included.
      The functions will be given one argument, the samples for a variable as an
      array, The functions should operate on an array, returning a single
      number. For example, `Statistics.mean`, or `Statistics.var` would both
      work.
- `extend::Bool=true`: If `true`, use the statistics returned by `stat_funcs` in
      addition to, rather than in place of, the default statistics. This is only
      meaningful when `stat_funcs` is not `nothing`.
- `credible_interval::Real=0.94`: Credible interval to plot. This is only
      meaningful when `stat_funcs` is `nothing`.
- `order::String="C"`: If `fmt` is "wide", use either "C" or "F" unpacking order.
- `index_origin::Int=1`: If `fmt` is "wide", select 𝑛-based indexing for
      multivariate parameters.

# Returns
- `Union{Pandas.DataFrame,Dataset}`: Return type dicated by `fmt` argument.
      Return value will contain summary statistics for each variable. Default
      statistics are:
    + `mean`
    + `sd`
    + `hpd_3%`
    + `hpd_97%`
    + `mcse_mean`
    + `mcse_sd`
    + `ess_bulk`
    + `ess_tail`
    + `r_hat` (only computed for traces with 2 or more chains)

# Examples

```@example summarystats
using ArviZ
idata = load_arviz_data("centered_eight")
summarystats(idata; var_names=["mu", "tau"])
```

Other statistics can be calculated by passing a list of functions or a
dictionary with key, function pairs:

```@example summarystats
using StatsBase, Statistics
function median_sd(x)
    med = median(x)
    sd = sqrt(mean((x .- med).^2))
    return sd
end

func_dict = Dict(
    "std" => x -> std(x; corrected = false),
    "median_std" => median_sd,
    "5%" => x -> percentile(x, 5),
    "median" => median,
    "95%" => x -> percentile(x, 95),
)

summarystats(idata; var_names = ["mu", "tau"], stat_funcs = func_dict, extend = false)
```
"""
function StatsBase.summarystats(data::Dataset; index_origin = 1, kwargs...)
    s = arviz.summary(data; index_origin = index_origin, kwargs...)
    s isa Dataset && return s
    return Pandas.DataFrame(s)
end

function StatsBase.summarystats(data::InferenceData; group = :posterior, kwargs...)
    dataset = getproperty(data, Symbol(group))
    return summarystats(dataset; kwargs...)
end

"""
    summary(
        data;
        group = :posterior,
        coords = nothing,
        dims = nothing,
        kwargs...,
    ) -> Union{Pandas.DataFrame,PyObject}

Compute summary statistics on any object that can be passed to
[`convert_to_dataset`](@ref).

# Keywords
- `coords::Dict{String,Vector}=nothing`: Map from named dimension to named
      indices.
- `dims::Dict{String,Vector{String}}=nothing`: Map from variable name to names
      of its dimensions.
- `kwargs`: Keyword arguments passed to [`summarystats`](@ref).
"""
function summary(data; group = :posterior, coords = nothing, dims = nothing, kwargs...)
    idata = convert_to_inference_data(data; group = group, coords = coords, dims = dims)
    return summarystats(idata; group = group, kwargs...)
end
