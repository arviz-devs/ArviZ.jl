compare(args...; kwargs...) = arviz.compare(args...; kwargs...) |> DataFrame

@forwardfun hpd

loo(args...; kwargs...) = arviz.loo(args...; kwargs...) |> Series

@forwardfun loo_pit

@forwardfun psislw

r2_score(args...; kwargs...) = arviz.r2_score(args...; kwargs...) |> Series

waic(args...; kwargs...) = arviz.waic(args...; kwargs...) |> Series

Base.summary(data::InferenceData) = arviz.summary(data) |> DataFrame
