"""
    compare(models; kwargs...)

Compare models based on their expected log pointwise predictive density (ELPD).

`models` is a `Tuple`, `NamedTuple`, or `AbstractVector` whose values are either
[`AbstractELPDResult`](@ref) entries or any argument to `elpd_method`, which must produce an
`AbstractELPDResult`.

The weights are returned in the same type of
collection.

The argument may be any object with a `pairs` method where each value is either an
[`InferenceData`](@ref) or an [`AbstractELPDResult`](@ref).

The ELPD is estimated either by Pareto smoothed importance sampling leave-one-out
cross-validation (LOO) or using the widely applicable information criterion (WAIC).
We recommend loo. Read more theory here - in a paper by some of the
leading authorities on model comparison dx.doi.org/10.1111/1467-9868.00353

# Arguments

  - `models`: a `Tuple`, `NamedTuple`, or `AbstractVector` whose values are either
    [`AbstractELPDResult`](@ref) entries or any argument to `elpd_method`.

# Keywords

  - `weights_method::AbstractModelWeightsMethod=Stacking()`: the method to be used to weight
    the models. See [`model_weights`](@ref) for details

      + `elpd_method=loo`: a method that computes an `AbstractELPDResult` from an argument in
        `models`.

  - `sort::Bool=true`: Whether to sort models by decreasing ELPD.

# Returns

  - [`ModelComparisonResult`](@ref): A container for the model comparison results.

# Examples

Compare the centered and non centered models of the eight school problem using the defaults:
[`loo`](@ref) and [`Stacking`](@ref) weights:

```jldoctest compare; filter = [r"└.*"]
julia> using ArviZ, ArviZExampleData

julia> models = (
           centered=load_example_data("centered_eight"),
           non_centered=load_example_data("non_centered_eight"),
       );

julia> mc = compare(models)
┌ Warning: 1 parameters had Pareto shape values 0.7 < k ≤ 1. Resulting importance sampling estimates are likely to be unstable.
└ @ PSIS ~/.julia/packages/PSIS/...
ModelComparisonResult with Stacking weights
 name          rank  elpd  elpd_mcse  elpd_diff  elpd_diff_mcse  weight    p   ⋯
 non_centered     1   -31        1.4       0              0.0       1.0  0.9   ⋯
 centered         2   -31        1.4       0.06           0.067     0.0  0.9   ⋯
                                                                1 column omitted
```

Compare the same models from pre-computed PSIS-LOO results and computing
[`BootstrappedPseudoBMA`](@ref) weights:

```jldoctest compare; setup = :(using Random; Random.seed!(23))
julia> elpd_results = mc.elpd_result;

julia> compare(elpd_results; weights_method=BootstrappedPseudoBMA())
ModelComparisonResult with BootstrappedPseudoBMA weights
 name          rank  elpd  elpd_mcse  elpd_diff  elpd_diff_mcse  weight    p   ⋯
 non_centered     1   -31        1.4       0              0.0      0.52  0.9   ⋯
 centered         2   -31        1.4       0.06           0.067    0.48  0.9   ⋯
                                                                1 column omitted
```
"""
function compare(
    inputs;
    weights_method::AbstractModelWeightsMethod=Stacking(),
    elpd_method=loo,
    model_names=_indices(inputs),
    sort::Bool=true,
)
    length(model_names) === length(inputs) ||
        throw(ArgumentError("Length of `model_names` must match length of `inputs`"))
    elpd_results = map(Base.Fix1(_maybe_elpd_results, elpd_method), inputs)
    weights = model_weights(weights_method, elpd_results)
    perm = _sortperm(elpd_results; by=x -> elpd_estimates(x).elpd, rev=true)
    i_elpd_max = first(perm)
    elpd_max_i = elpd_estimates(elpd_results[i_elpd_max]; pointwise=true).elpd
    elpd_diff_and_mcse = map(elpd_results) do r
        elpd_diff_j = similar(elpd_max_i)
        # workaround for named dimension packages that check dimension names are exact, for
        # cases where dimension names differ
        map!(-, elpd_diff_j, elpd_max_i, elpd_estimates(r; pointwise=true).elpd)
        return _sum_and_se(elpd_diff_j)
    end
    elpd_diff = map(first, elpd_diff_and_mcse)
    elpd_diff_mcse = map(last, elpd_diff_and_mcse)
    rank = _assimilar(elpd_results, (1:length(elpd_results))[perm])
    result = ModelComparisonResult(
        model_names, rank, elpd_diff, elpd_diff_mcse, weights, elpd_results, weights_method
    )
    sort || return result
    return _permute(result, perm)
end

_maybe_elpd_results(elpd_method, x::AbstractELPDResult; kwargs...) = x
function _maybe_elpd_results(elpd_method, x; kwargs...)
    elpd_result = elpd_method(x; kwargs...)
    elpd_result isa AbstractELPDResult && return elpd_result
    throw(
        ErrorException(
            "Return value of `elpd_method` must be an `AbstractELPDResult`, not `$(typeof(elpd_result))`.",
        ),
    )
end

"""
    ModelComparisonResult

Result of model comparison using ELPD.

This struct implements the Tables and TableTraits interfaces.

Each field returns a collection of the corresponding entry for each model:
$(FIELDS)
"""
struct ModelComparisonResult{E,N,R,W,ER,M}
    "Names of the models, if provided."
    name::N
    "Ranks of the models (ordered by decreasing ELPD)"
    rank::R
    "ELPD of a model subtracted from the largest ELPD of any model"
    elpd_diff::E
    "Monte Carlo standard error of the ELPD difference"
    elpd_diff_mcse::E
    "Model weights computed with `weights_method`"
    weight::W
    """`AbstactELPDResult`s for each model, which can be used to access useful stats like
    ELPD estimates, pointwise estimates, and Pareto shape values for PSIS-LOO"""
    elpd_result::ER
    "Method used to compute model weights with [`model_weights`](@ref)"
    weights_method::M
end

function Base.show(io::IO, ::MIME"text/plain", r::ModelComparisonResult; sigdigits_se=2)
    weights_method_name = _typename(r.weights_method)

    table = Tables.columntable(r)
    cols = Tables.columnnames(table)

    # formatting for columns
    est_cols = findall(∈((:elpd, :elpd_diff, :p)), cols)
    se_cols = findall(∈((:elpd_mcse, :elpd_diff_mcse, :p_mcse)), cols)
    est_formatters = map(est_cols, se_cols) do est_col, se_col
        ft_printf_sigdigits_matching_se(table[se_col], [est_col])
    end
    se_formatter = ft_printf_sigdigits(sigdigits_se, se_cols)
    weights = table.weight
    digits_weights = ceil(Int, -log10(maximum(weights))) + 1
    weight_formatter = PrettyTables.ft_printf(
        "%.$(digits_weights)f", findfirst(==(:weight), cols)
    )
    formatters = (est_formatters..., se_formatter, weight_formatter)

    alignment_anchor_regex = Dict(
        i => [r"\.", r"e", r"^NaN$", r"Inf$"] for
        (i, (k, v)) in enumerate(pairs(table)) if (eltype(v) <: Real)
    )
    alignment = [:l, fill(:r, length(cols) - 1)...]
    alignment_anchor_fallback = :r

    PrettyTables.pretty_table(
        io,
        table;
        title="ModelComparisonResult with $(weights_method_name) weights",
        title_crayon=PrettyTables.Crayon(),
        show_subheader=false,
        hlines=:none,
        vlines=:none,
        newline_at_end=false,
        formatters,
        alignment_anchor_regex,
        alignment,
        alignment_anchor_fallback,
    )
    return nothing
end

function _permute(r::ModelComparisonResult, perm)
    return ModelComparisonResult(
        (_permute(getfield(r, k), perm) for k in fieldnames(typeof(r))[1:(end - 1)])...,
        r.weights_method,
    )
end

#### Tables interface as column table

Tables.istable(::Type{<:ModelComparisonResult}) = true
Tables.columnaccess(::Type{<:ModelComparisonResult}) = true
Tables.columns(r::ModelComparisonResult) = r
function Tables.columnnames(::ModelComparisonResult)
    return (
        :name, :rank, :elpd, :elpd_mcse, :elpd_diff, :elpd_diff_mcse, :weight, :p, :p_mcse
    )
end
function Tables.getcolumn(r::ModelComparisonResult, i::Int)
    return Tables.getcolumn(r, Tables.columnnames(r)[i])
end
function Tables.getcolumn(r::ModelComparisonResult, nm::Symbol)
    nm ∈ fieldnames(typeof(r)) && return getfield(r, nm)
    if nm ∈ (:elpd, :elpd_mcse, :p, :p_mcse)
        return map(e -> getproperty(elpd_estimates(e), nm), r.elpd_result)
    end
    throw(ArgumentError("Unrecognized column name $nm"))
end

IteratorInterfaceExtensions.isiterable(::ModelComparisonResult) = true
function IteratorInterfaceExtensions.getiterator(r::ModelComparisonResult)
    return Tables.datavaluerows(Tables.columntable(r))
end

TableTraits.isiterabletable(::ModelComparisonResult) = true
