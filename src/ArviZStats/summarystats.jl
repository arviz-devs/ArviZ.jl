const DEFAULT_METRIC_DIM = Dimensions.key2dim(:_metric)

"""
$(SIGNATURES)

A container for a column table of values computed by [`summarystats`](@ref).

This object implements the Tables and TableTraits interfaces and has a custom `show` method.

$(FIELDS)
"""
struct SummaryStats{D<:NamedTuple}
    """The summary statistics for each variable, with the first entry containing the
    variable names"""
    data::D
end

# forward key interfaces from its parent
Base.parent(stats::SummaryStats) = getfield(stats, :data)
Base.propertynames(stats::SummaryStats) = propertynames(parent(stats))
Base.getproperty(stats::SummaryStats, nm::Symbol) = getproperty(parent(stats), nm)
Base.keys(stats::SummaryStats) = keys(parent(stats))
Base.haskey(stats::SummaryStats, nm::Symbol) = haskey(parent(stats), nm)
Base.length(stats::SummaryStats) = length(parent(stats))
Base.getindex(stats::SummaryStats, i::Int) = getindex(parent(stats), i)
Base.getindex(stats::SummaryStats, nm::Symbol) = getindex(parent(stats), nm)
function Base.iterate(stats::SummaryStats, i::Int=firstindex(parent(stats)))
    return iterate(parent(stats), i)
end

#### custom tabular show methods

function Base.show(io::IO, mime::MIME"text/plain", stats::SummaryStats; kwargs...)
    return _show(io, mime, stats; kwargs...)
end
function Base.show(io::IO, mime::MIME"text/html", stats::SummaryStats; kwargs...)
    return _show(io, mime, stats; kwargs...)
end

function _show(io::IO, mime::MIME, stats::SummaryStats; kwargs...)
    data = parent(stats)[eachindex(stats)[2:end]]
    rhat_formatter = _prettytables_rhat_formatter(data)
    extra_formatters = rhat_formatter === nothing ? () : (rhat_formatter,)
    return _show_prettytable(
        io,
        mime,
        data;
        title="SummaryStats",
        row_labels=stats.variable,
        extra_formatters,
        kwargs...,
    )
end

#### Tables interface as column table

Tables.istable(::Type{<:SummaryStats}) = true
Tables.columnaccess(::Type{<:SummaryStats}) = true
Tables.columns(s::SummaryStats) = s
Tables.columnnames(s::SummaryStats) = Tables.columnnames(parent(s))
Tables.getcolumn(s::SummaryStats, i::Int) = Tables.getcolumn(parent(s), i)
Tables.getcolumn(s::SummaryStats, nm::Symbol) = Tables.getcolumn(parent(s), nm)

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

  - `prob_interval::Real`: The value of the `prob` argument to [`hdi`](@ref) used to compute
    the highest density interval. Defaults to $(DEFAULT_INTERVAL_PROB).
  - `return_type::Type`: The type of object to return. Valid options are [`Dataset`](@ref)
    and [`SummaryStats`](@ref). Defaults to `SummaryStats`.
  - `metric_dim`: The dimension name or type to use for the computed metrics. Only used
    if `return_type` is `Dataset`. Defaults to `$(sprint(show, "text/plain", DEFAULT_METRIC_DIM))`.
  - `compact_labels::Bool`: Whether to use compact names for the variables. Only used if
    `return_type` is `SummaryStats`. Defaults to `true`.
  - `kind::Symbol`: Whether to compute just statistics (`:stats`), just diagnostics
    (`:diagnostics`), or both (`:both`). Defaults to `:both`.

# Examples

Compute the summary statistics and diagnostics on posterior draws of the centered eight
model:

```jldoctest summarystats
julia> using ArviZ, ArviZExampleData

julia> idata = load_example_data("centered_eight");

julia> summarystats(idata.posterior[(:mu, :tau)])
SummaryStats
      mean  std  hdi_3%  hdi_97%  mcse_mean  mcse_std  ess_tail  ess_bulk  rha ⋯
 mu    4.5  3.5  -1.62     10.7        0.23      0.11       659       241  1.0 ⋯
 tau   4.1  3.1   0.896     9.67       0.26      0.17        38        67  1.0 ⋯
                                                                1 column omitted
```

Compute just the statistics on all variables:

```jldoctest summarystats
julia> summarystats(idata.posterior; kind=:stats)
SummaryStats
                          mean   std  hdi_3%  hdi_97%
 mu                       4.49  3.49  -1.62     10.7
 theta[Choate]            6.46  5.87  -4.56     17.1
 theta[Deerfield]         5.03  4.88  -4.31     14.3
 theta[Phillips Andover]  3.94  5.69  -7.77     13.7
 theta[Phillips Exeter]   4.87  5.01  -4.49     14.7
 theta[Hotchkiss]         3.67  4.96  -6.47     11.7
 theta[Lawrenceville]     3.97  5.19  -7.04     12.2
 theta[St. Paul's]        6.58  5.11  -3.09     16.3
 theta[Mt. Hermon]        4.77  5.74  -5.86     16.0
 tau                      4.12  3.10   0.896     9.67
```

Compute the statistics and diagnostics from the posterior group of an `InferenceData` and
store in a `Dataset`:

```jldoctest summarystats
julia> summarystats(idata; return_type=Dataset)
Dataset with dimensions:
  Dim{:_metric} Categorical{String} String[mean, std, …, ess_bulk, rhat] Unordered,
  Dim{:school} Categorical{String} String[Choate, Deerfield, …, St. Paul's, Mt. Hermon] Unordered
and 3 layers:
  :mu    Float64 dims: Dim{:_metric} (9)
  :theta Float64 dims: Dim{:school}, Dim{:_metric} (8×9)
  :tau   Float64 dims: Dim{:_metric} (9)
```
"""
function StatsBase.summarystats(
    data::InferenceObjects.InferenceData; group::Symbol=:posterior, kwargs...
)
    return summarystats(data[group]; kwargs...)
end
function StatsBase.summarystats(
    data::InferenceObjects.Dataset; return_type::Type=SummaryStats, kwargs...
)
    return _summarize(return_type, data; kwargs...)
end

function _summarize(
    ::Type{InferenceObjects.Dataset},
    data::InferenceObjects.Dataset;
    kind::Symbol=:both,
    prob_interval::Real=DEFAULT_INTERVAL_PROB,
    metric_dim=DEFAULT_METRIC_DIM,
)
    dims = Dimensions.dims(data, InferenceObjects.DEFAULT_SAMPLE_DIMS)
    stats = [
        "mean" => (data -> dropdims(Statistics.mean(data; dims); dims)),
        "std" => (data -> dropdims(Statistics.std(data; dims); dims)),
        _interval_prob_to_strings("hdi", prob_interval) =>
            (data -> hdi(data; prob=prob_interval)),
    ]
    diagnostics = [
        "mcse_mean" => (data -> MCMCDiagnosticTools.mcse(data; kind=Statistics.mean)),
        "mcse_std" => (data -> MCMCDiagnosticTools.mcse(data; kind=Statistics.std)),
        "ess_tail" => (data -> MCMCDiagnosticTools.ess(data; kind=:tail)),
        ("ess_bulk", "rhat") => (data -> MCMCDiagnosticTools.ess_rhat(data)),
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
    metric_names = collect(Iterators.flatten(map(_astuple ∘ first, metrics)))
    cat_dim = Dimensions.rebuild(Dimensions.basedims(metric_dim), metric_names)
    ds = cat(metric_vals...; dims=cat_dim)::InferenceObjects.Dataset
    return DimensionalData.rebuild(ds; metadata=DimensionalData.NoMetadata(), refdims=dims)
end
function _summarize(
    ::Type{SummaryStats},
    data::InferenceObjects.Dataset;
    compact_labels::Bool=true,
    metric_dim=DEFAULT_METRIC_DIM,
    kwargs...,
)
    ds = _summarize(InferenceObjects.Dataset, data; metric_dim, kwargs...)
    table = _as_flat_table(ds, metric_dim; compact_labels)
    return SummaryStats(table)
end

function _interval_prob_to_strings(interval_type, prob; digits=2)
    α = (1 - prob) / 2
    perc_lower = string(round(100 * α; digits))
    perc_upper = string(round(100 * (1 - α); digits))
    return map((perc_lower, perc_upper)) do s
        s = replace(s, r"\.0+$" => "")
        return "$(interval_type)_$s%"
    end
end

function _as_flat_table(ds, dim; compact_labels::Bool=true)
    row_table = Iterators.map(_indices_iterator(ds, dim)) do (var, indices)
        var_select = isempty(indices) ? var : view(var, indices...)
        return (
            variable=_indices_to_name(var, indices, compact_labels),
            _arr_to_namedtuple(var_select)...,
        )
    end
    return Tables.columntable(row_table)
end

function _indices_iterator(ds::DimensionalData.AbstractDimStack, dims)
    return Iterators.flatten(
        Iterators.map(Base.Fix2(_indices_iterator, dims), DimensionalData.layers(ds))
    )
end
function _indices_iterator(var::DimensionalData.AbstractDimArray, dims)
    dims_flatten = Dimensions.otherdims(var, dims)
    isempty(dims_flatten) && return ((var, ()),)
    indices_iter = DimensionalData.DimKeys(dims_flatten)
    return zip(Iterators.cycle((var,)), indices_iter)
end

function _arr_to_namedtuple(arr::DimensionalData.AbstractDimVector)
    ks = Tuple(map(Symbol, DimensionalData.lookup(arr, 1)))
    return NamedTuple{ks}(Tuple(arr))
end

function _indices_to_name(var, dims, compact)
    name = DimensionalData.name(var)
    isempty(dims) && return string(name)
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
