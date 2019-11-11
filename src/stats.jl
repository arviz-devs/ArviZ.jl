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

compare(args...; kwargs...) = arviz.compare(args...; kwargs...) |> Pandas.DataFrame

Base.Docs.getdoc(::typeof(compare)) = Base.Docs.getdoc(arviz.compare)

@forwardfun hpd

loo(args...; kwargs...) = arviz.loo(args...; kwargs...) |> Pandas.Series

Base.Docs.getdoc(::typeof(loo)) = Base.Docs.getdoc(arviz.loo)

@forwardfun loo_pit

@forwardfun psislw

r2_score(args...; kwargs...) = arviz.r2_score(args...; kwargs...) |> Pandas.Series

Base.Docs.getdoc(::typeof(r2_score)) = Base.Docs.getdoc(arviz.r2_score)

waic(args...; kwargs...) = arviz.waic(args...; kwargs...) |> Pandas.Series

Base.Docs.getdoc(::typeof(waic)) = Base.Docs.getdoc(arviz.waic)

Base.summary(data::InferenceData) = arviz.summary(data) |> Pandas.DataFrame

Base.Docs.getdoc(::typeof(summary)) = Base.Docs.getdoc(arviz.summary)
