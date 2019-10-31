compare(args...; kwargs...) = arviz.compare(args...; kwargs...) |> DataFrame

Base.Docs.getdoc(::typeof(compare)) = Base.Docs.getdoc(arviz.compare)

@forwardfun hpd

loo(args...; kwargs...) = arviz.loo(args...; kwargs...) |> Series

Base.Docs.getdoc(::typeof(loo)) = Base.Docs.getdoc(arviz.loo)

@forwardfun loo_pit

@forwardfun psislw

r2_score(args...; kwargs...) = arviz.r2_score(args...; kwargs...) |> Series

Base.Docs.getdoc(::typeof(r2_score)) = Base.Docs.getdoc(arviz.r2_score)

waic(args...; kwargs...) = arviz.waic(args...; kwargs...) |> Series

Base.Docs.getdoc(::typeof(waic)) = Base.Docs.getdoc(arviz.waic)

Base.summary(data::InferenceData) = arviz.summary(data) |> DataFrame

Base.Docs.getdoc(::typeof(summary)) = Base.Docs.getdoc(arviz.summary)
