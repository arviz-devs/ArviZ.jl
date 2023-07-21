# [Stats](@id stats-api)

```@index
Pages = ["stats.md"]
```

## General statistics

```@docs
hdi
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
ArviZStats.smooth_data
```
