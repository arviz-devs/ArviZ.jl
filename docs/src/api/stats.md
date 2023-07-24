# [Stats](@id stats-api)

```@index
Pages = ["stats.md"]
```

## General statistics

```@docs
hdi
hdi!
ArviZ.summary
summarystats
r2_score
```

## Pareto-smoothed importance sampling

```@docs
PSIS.PSISResult
PSIS.psis
PSIS.psis!
```

## Model assessment and selection

### LOO and WAIC

```@docs
AbstractELPDResult
PSISLOOResult
WAICResult
elpd_estimates
information_criterion
loo
waic
```

### Others

```@docs
compare
loo_pit
```

### Utilities

```@docs
ArviZStats.smooth_data
```
