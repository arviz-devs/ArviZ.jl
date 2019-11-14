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

@doc forwarddoc(:compare)
compare(args...; kwargs...) = arviz.compare(args...; kwargs...) |> Pandas.DataFrame

Docs.getdoc(::typeof(compare)) = forwardgetdoc(:compare)

@forwardfun hpd

@doc forwarddoc(:loo)
loo(args...; kwargs...) = arviz.loo(args...; kwargs...) |> Pandas.Series

Docs.getdoc(::typeof(loo)) = forwardgetdoc(:loo)

@forwardfun loo_pit

@forwardfun psislw

@doc forwarddoc(:r2_score)
r2_score(args...; kwargs...) = arviz.r2_score(args...; kwargs...) |> Pandas.Series

Docs.getdoc(::typeof(r2_score)) = forwardgetdoc(:r2_score)

@doc forwarddoc(:waic)
waic(args...; kwargs...) = arviz.waic(args...; kwargs...) |> Pandas.Series

Docs.getdoc(::typeof(waic)) = forwardgetdoc(:waic)

@doc forwarddoc(:summary)
Base.summary(data::InferenceData) = arviz.summary(data) |> Pandas.DataFrame
