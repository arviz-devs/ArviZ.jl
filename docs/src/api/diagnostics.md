# [Diagnostics](@id diagnostics-api)

```@index
Pages = ["diagnostics.md"]
```

## [Bayesian fraction of missing information](@id bfmi)

```@docs
MCMCDiagnosticTools.bfmi
```

## [Effective sample size and $\widehat{R}$ diagnostic](@id ess_rhat)

```@docs
MCMCDiagnosticTools.ess
MCMCDiagnosticTools.rhat
MCMCDiagnosticTools.ess_rhat
```

The following autocovariance methods are supported:

```@docs
MCMCDiagnosticTools.AutocovMethod
MCMCDiagnosticTools.FFTAutocovMethod
MCMCDiagnosticTools.BDAAutocovMethod
```

## [Monte Carlo standard error](@id mcse)

```@docs
MCMCDiagnosticTools.mcse
```

## [$R^*$ diagnostic](@id rstar)

```@docs
MCMCDiagnosticTools.rstar
```
