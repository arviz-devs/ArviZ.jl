# [Stats](@id stats-api)

```@index
Pages = ["stats.md"]
```

## Summary statistics

```@docs
SummaryStats
default_summary_stats
summarize
summarystats
```

## Credible intervals

```@docs
eti
eti!
hdi
hdi!
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
r2_score
```

## Utilities

```@docs
PosteriorStats.kde_reflected
PosteriorStats.pointwise_loglikelihoods
```
