"""
    SummaryStats{D}

A container for a column table of values computed by [`summarystatistics`](@ref).

The first column is `variable`, and all remaining columns are the summary statistics.
This object implements the Tables and TableTraits interfaces and has a custom `show` method.
"""
struct SummaryStats{D}
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

function Base.show(
    io::IO,
    ::MIME"text/plain",
    stats::SummaryStats;
    sigdigits_se=2,
    sigdigits_default=3,
    kwargs...,
)
    # formatting functions for special columns
    # see https://ronisbr.github.io/PrettyTables.jl/stable/man/formatters/
    formatters = []
    for (i, k) in enumerate(keys(stats))
        mcse_key = Symbol("mcse_$k")
        if haskey(stats, mcse_key)
            push!(formatters, ft_printf_sigdigits_matching_se(stats[mcse_key], [i]))
        end
    end
    mcse_cols = findall(startswith("mcse_") ∘ string, keys(stats))
    isempty(mcse_cols) || push!(formatters, ft_printf_sigdigits(sigdigits_se, mcse_cols))
    ess_cols = findall(_is_ess_label, keys(stats))
    isempty(ess_cols) || push!(formatters, PrettyTables.ft_printf("%d", ess_cols))
    if haskey(stats, :rhat)
        push!(formatters, PrettyTables.ft_printf("%.2f", findfirst(==(:rhat), keys(stats))))
    end
    push!(formatters, ft_printf_sigdigits(sigdigits_default))

    # align by decimal point exponent, or beginning of Inf or NaN, if present.
    # Otherwise, right align, except for variable names
    # ESS values are special-cased to always be right-aligned, even if Infs or NaNs are present
    alignment = [:l, fill(:r, length(stats) - 1)...]
    alignment_anchor_regex = Dict(
        i => [r"\.", r"e", r"^NaN$", r"Inf$"] for
        (i, (k, v)) in enumerate(pairs(stats)) if (eltype(v) <: Real && !_is_ess_label(k))
    )
    alignment_anchor_fallback = :r
    alignment_anchor_fallback_override = Dict(
        i => :r for (i, k) in enumerate(keys(stats)) if _is_ess_label(k)
    )

    # TODO: highlight bad values in the REPL

    kwargs_new = merge(
        (
            title="SummaryStats",
            title_crayon=PrettyTables.Crayon(),
            header=["", Iterators.drop(keys(stats), 1)...],  # drop "variable" from header
            show_subheader=false,
            hlines=:none,
            vlines=:none,
            newline_at_end=false,
            formatters=Tuple(formatters),
            alignment,
            alignment_anchor_regex,
            alignment_anchor_fallback,
            alignment_anchor_fallback_override,
        ),
        kwargs,
    )
    PrettyTables.pretty_table(io, stats; kwargs_new...)

    return nothing
end

_is_ess_label(k::Symbol) = ((k === :ess) || startswith(string(k), "ess_"))

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

  - `return_type::Type`: The type of object to return. Valid options are [`Dataset`](@ref)
  and [`SummaryStats`](@ref). Defaults to `SummaryStats`.
  - `prob_interval::Real`: The value of the `prob` argument to [`hdi`](@ref) used to compute the
  highest density interval. Defaults to $(DEFAULT_INTERVAL_PROB).
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
    metric_dim=Dimensions.Dim{:metric},
)
    dims = InferenceObjects.DEFAULT_SAMPLE_DIMS
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

function _interval_prob_to_strings(interval_type, prob; digits=2)
    α = (1 - prob) / 2
    perc_lower = string(round(100 * α; digits))
    perc_upper = string(round(100 * (1 - α); digits))
    return map((perc_lower, perc_upper)) do s
        s = replace(s, r"\.0+$" => "")
        return "$(interval_type)_$s%"
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
