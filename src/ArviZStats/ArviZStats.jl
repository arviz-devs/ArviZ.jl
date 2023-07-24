module ArviZStats

using ArviZ: ArviZ, arviz, @forwardfun
using DataInterpolations: DataInterpolations
using DimensionalData: DimensionalData, Dimensions
using Distributions: Distributions
using DocStringExtensions: FIELDS, FUNCTIONNAME, TYPEDEF, TYPEDFIELDS, SIGNATURES
using InferenceObjects: InferenceObjects
using IteratorInterfaceExtensions: IteratorInterfaceExtensions
using LinearAlgebra: mul!, norm
using LogExpFunctions: LogExpFunctions
using Markdown: @doc_str
using MCMCDiagnosticTools: MCMCDiagnosticTools
using Optim: Optim
using PrettyTables: PrettyTables
using Printf: Printf
using PSIS: PSIS, PSISResult, psis, psis!
using PyCall: PyCall
using Random: Random
using Setfield: Setfield
using Statistics: Statistics
using StatsBase: StatsBase, summarystats
using Tables: Tables
using TableTraits: TableTraits

# PSIS
export PSIS, PSISResult, psis, psis!

# LOO-CV
export AbstractELPDResult, PSISLOOResult, WAICResult
export elpd_estimates, information_criterion, loo, waic

# Model weighting and comparison
export AbstractModelWeightsMethod, BootstrappedPseudoBMA, PseudoBMA, Stacking, model_weights
export ModelComparisonResult, compare

# Others
export hdi, kde, loo_pit, r2_score, summary, summarystats

# load for docstrings
using ArviZ: InferenceData, convert_to_dataset, ess

const INFORMATION_CRITERION_SCALES = (deviance=-2, log=1, negative_log=-1)

@forwardfun hdi
@forwardfun kde
@forwardfun r2_score

include("utils.jl")
include("elpdresult.jl")
include("loo.jl")
include("waic.jl")
include("model_weights.jl")
include("compare.jl")
include("loo_pit.jl")

ArviZ.convert_result(::typeof(r2_score), result) = ArviZ.todataframes(result)

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
using ArviZ, ArviZExampleData
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
function StatsBase.summarystats(
    data::InferenceObjects.InferenceData; group::Symbol=:posterior, kwargs...
)
    dataset = getproperty(data, group)
    return StatsBase.summarystats(dataset; kwargs...)
end
function StatsBase.summarystats(
    data::InferenceObjects.Dataset; var_names=nothing, digits::Int=typemax(Int), kwargs...
)
    var_names = var_names === nothing ? var_names : collect(var_names)
    round_to = digits == typemax(Int) ? nothing : digits
    s = arviz.summary(data; var_names, round_to, kwargs...)
    return ArviZ.todataframes(s; index_name=:variable)
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
    dataset = InferenceObjects.convert_to_dataset(data; group, coords, dims)
    return StatsBase.summarystats(dataset; kwargs...)
end

end  # module
