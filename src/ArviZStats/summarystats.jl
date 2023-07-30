"""
    SummaryStats{D}

A container for a column table of values computed by [`summarystatistics`](@ref).

The first column is `variable`, and all remaining columns are the summary statistics.
This object implements the Tables and TableTraits interfaces and has a custom `show` method.
"""
struct SummaryStats{D}
    data::D
end

function Base.show(
    io::IO, stats::SummaryStats; sigdigits_se=2, sigdigits_default=3, kwargs...
)
    colnames = keys(stats.data)
    formatters = function (v, i, j)
        colname = colnames[j]
        if j == 1
            return v
        elseif Symbol("mcse_$colname") ∈ colnames
            se = stats.data[Symbol("mcse_$colname")][i]
            sigdigits = sigdigits_matching_error(v, se)
            sprint(Printf.format, Printf.Format("%.$(sigdigits)g"), v)
        elseif startswith(string(colname), "mcse")
            sigdigits = sigdigits_se
            return sprint(Printf.format, Printf.Format("%.$(sigdigits)g"), v)
        elseif colname === :rhat
            return sprint(Printf.format, Printf.Format("%.2f"), v)
        elseif startswith(string(colname), "ess")
            return sprint(Printf.format, Printf.Format("%d"), v)
        else
            return sprint(Printf.format, Printf.Format("%.$(sigdigits_default)g"), v)
        end
    end
    kwargs_new = merge(
        (
            title="SummaryStatistics",
            show_subheader=false,
            hlines=:none,
            vlines=:none,
            formatters,
            newline_at_end=false,
            alignment=[:l, fill(:r, length(colnames) - 1)...],
            header=["", colnames[2:end]...],
            title_crayon=PrettyTables.Crayon(),
        ),
        kwargs,
    )
    PrettyTables.pretty_table(io, stats.data; kwargs_new...)
    return nothing
end

#### Tables interface as column table

Tables.istable(::Type{<:SummaryStats}) = true
Tables.columnaccess(::Type{<:SummaryStats}) = true
Tables.columns(s::SummaryStats) = s
Tables.columnnames(s::SummaryStats) = Tables.columnnames(s.data)
Tables.getcolumn(s::SummaryStats, i::Int) = Tables.getcolumn(s.data, i)
Tables.getcolumn(s::SummaryStats, nm::Symbol) = Tables.getcolumn(s.data, nm)

IteratorInterfaceExtensions.isiterable(::SummaryStats) = true
function IteratorInterfaceExtensions.getiterator(s::SummaryStats)
    return Tables.datavaluerows(Tables.columntable(s))
end

TableTraits.isiterabletable(::SummaryStats) = true

"""
    summarystats(data::InferenceData; group=:posterior, kwargs...)
    summarystats(data::Dataset; kwargs...)

Compute summary statistics and diagnostics on the `data`.

# Keywords

  - `return_type::Type`: The type of object to return. Valid options are [`Dataset`](@ref)
  and [`SummaryStats`](@ref). Defaults to `SummaryStats`.
  - `hdi_prob::Real`: The value of the `prob` argument to [`hdi`](@ref) used to compute the
  highest density interval. Defaults to $(HDI_DEFAULT_PROB).
  - `metric_dim`: The dimension name or type to use for the computed metrics. Only specify
  if `return_type` is `Dataset`. Defaults to `Dim{_:metric}`.
  - `compact_names::Bool`: Whether to use compact names for the variables. Only used if
  `return_type` is `SummaryStats`. Defaults to `true`.
  - `kind::Symbol`: Whether to compute just statistics (`:stats`), just diagnostics
  (`:diagnostics`), or both (`:both`). Defaults to `:both`.

# Examples

Compute the summary statistics and diagnostics on posterior draws of the centered eight
model:

```jldoctest summarystats
julia> idata = load_example_data("centered_eight");

julia> summarystats(idata.posterior[(:mu, :tau)])
SummaryStatistics
      mean  std  hdi_3%  hdi_97%  mcse_mean  mcse_std  ess_tail  ess_bulk  rhat
 mu    4.5  3.5   -1.62     10.7       0.23      0.11       659       241  1.02
 tau   4.1  3.1   0.896     9.67       0.26      0.17        38        67  1.06
```

Compute just the statistics on all variables:

```jldoctest summarystats
julia> summarystats(idata.posterior; kind=:stats)
SummaryStatistics
                          mean   std  hdi_3%  hdi_97%
 mu                       4.49  3.49   -1.62     10.7
 theta[Choate]            6.46  5.87   -4.56     17.1
 theta[Deerfield]         5.03  4.88   -4.31     14.3
 theta[Phillips Andover]  3.94  5.69   -7.77     13.7
 theta[Phillips Exeter]   4.87  5.01   -4.49     14.7
 theta[Hotchkiss]         3.67  4.96   -6.47     11.7
 theta[Lawrenceville]     3.97  5.19   -7.04     12.2
 theta[St. Paul's]        6.58  5.11   -3.09     16.3
 theta[Mt. Hermon]        4.77  5.74   -5.86       16
 tau                      4.12   3.1   0.896     9.67
```
"""
function summarystats(
    data::InferenceObjects.InferenceData; group::Symbol=:posterior, kwargs...
)
    return summarystats(data[group]; kwargs...)
end
function summarystats(
    data::InferenceObjects.Dataset; return_type::Type=SummaryStats, kwargs...
)
    return _summarize(return_type, data; kwargs...)
end

function _summarize(
    ::Type{InferenceObjects.Dataset},
    data::InferenceObjects.Dataset;
    kind::Symbol=:both,
    hdi_prob::Real=HDI_DEFAULT_PROB,
    metric_dim=Dimensions.Dim{:metric},
)
    dims = InferenceObjects.DEFAULT_SAMPLE_DIMS
    stats = [
        "mean" => (data -> dropdims(Statistics.mean(data; dims); dims)),
        "std" => (data -> dropdims(Statistics.std(data; dims); dims)),
        _hdi_prob_to_strings(hdi_prob) => (data -> hdi(data; prob=hdi_prob)),
    ]
    diagnostics = [
        "mcse_mean" => (data -> MCMCDiagnosticTools.mcse(data; kind=Statistics.mean)),
        "mcse_std" => (data -> MCMCDiagnosticTools.mcse(data; kind=Statistics.std)),
        "ess_tail" => (data -> MCMCDiagnosticTools.ess(data; kind=:tail)),
        ("ess_bulk", "rhat") => (data -> MCMCDiagnosticTools.ess_rhat(data; kind=:bulk)),
    ]
    metrics = if kind === :both
        vcat(stats, diagnostics)
    elseif kind === :stats
        stats
    elseif kind === :diagnostics
        diagnostics
    else
        error("Invalid value for `kind`: $kind")
    end
    metric_vals = map(metrics) do (_, f)
        f(data)
    end
    metric_names = collect(Iterators.flatten(vcat(map(_astuple ∘ first, metrics))))
    cat_dim = metric_dim(metric_names)
    return cat(metric_vals...; dims=cat_dim)::InferenceObjects.Dataset
end
function _summarize(
    ::Type{SummaryStats},
    data::InferenceObjects.Dataset;
    compact_names::Bool=true,
    kwargs...,
)
    metric_dim = Dimensions.Dim{:_metric}
    ds = _summarize(InferenceObjects.Dataset, data; metric_dim, kwargs...)
    row_iter = _flat_iterator(ds, metric_dim; compact_names)
    nts = Tables.columntable(row_iter)
    return SummaryStats(nts)
end

function _hdi_prob_to_strings(prob; digits=2)
    α = (1 - prob) / 2
    perc_lower = string(round(100 * α; digits))
    perc_upper = string(round(100 * (1 - α); digits))
    return map((perc_lower, perc_upper)) do s
        s = replace(s, r"\.0+$" => "")
        return "hdi_$s%"
    end
end

function _flat_iterator(ds, dim; compact_names=true)
    var_iter = pairs(DimensionalData.layers(ds))
    return Iterators.flatten(
        Iterators.map(var_iter) do (var_name, var)
            dims_flatten = Dimensions.otherdims(var, dim)
            isempty(dims_flatten) &&
                return ((variable="$var_name", _arr_to_namedtuple(var)...),)
            indices_iter = DimensionalData.DimKeys(dims_flatten)
            return Iterators.map(indices_iter) do indices
                (
                    variable=_indices_to_name(var_name, indices, compact_names),
                    _arr_to_namedtuple(view(var, indices...))...,
                )
            end
        end,
    )
end

function _arr_to_namedtuple(arr::DimensionalData.AbstractDimVector)
    ks = Tuple(map(Symbol, DimensionalData.lookup(arr, 1)))
    return NamedTuple{ks}(Tuple(arr))
end

function _indices_to_name(name, dims, compact)
    elements = if compact
        map(string ∘ Dimensions.val ∘ Dimensions.val, dims)
    else
        map(dims) do d
            val = Dimensions.val(Dimensions.val(d))
            val_str = sprint(show, "text/plain", val)
            return "$(Dimensions.name(d))=At($val_str)"
        end
    end
    return "$name[" * join(elements, ',') * "]"
end

"""
    summarystats(data::InferenceData; group=:posterior, kwargs...)
    summarystats(data::Dataset; kwargs...)

Compute summary statistics on `data`.

These methods simply forward to [`summarize`](@ref). See that docstring for details.
"""
function StatsBase.summarystats(data::InferenceObjects.InferenceData; kwargs...)
    return summarize(data; kwargs...)
end
function StatsBase.summarystats(data::InferenceObjects.Dataset; kwargs...)
    return summarize(data; kwargs...)
end

