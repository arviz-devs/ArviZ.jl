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

## Predictive checks

```@docs
loo_pit
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

### Utilities

```@docs
ArviZStats.smooth_data
```
