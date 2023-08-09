# [Stats](@id stats-api)

```@index
Pages = ["stats.md"]
```

## Summary statistics

```@docs
SummaryStats
default_summary_stats
default_stats
default_diagnostics
summarize
summarystats
```

## General statistics

```@docs
hdi
hdi!
r2_score
```

## Pareto-smoothed importance sampling

```@docs
PSISResult
ess_is
PSISPlots.paretoshapeplot
psis
psis!
```

## LOO and WAIC

```@docs
AbstractELPDResult
PSISLOOResult
WAICResult
elpd_estimates
information_criterion
loo
waic
```

## Model comparison

```@docs
ModelComparisonResult
compare
model_weights
```

The following model weighting methods are available
```@docs
AbstractModelWeightsMethod
BootstrappedPseudoBMA
PseudoBMA
Stacking
```

## Predictive checks

```@docs
loo_pit
```

## Utilities

```@docs
PosteriorStats.smooth_data
```
