# [ArviZ.jl: Exploratory analysis of Bayesian models in Julia](@id arvizjl)

ArviZ.jl is a Julia meta-package for exploratory analysis of Bayesian models.
It is part of the [ArviZ project](https://www.arviz.org/), which also includes a related [Python package](https://python.arviz.org/).

ArviZ consists of and re-exports the following subpackages, along with extensions integrating them with InferenceObjects:
- [InferenceObjects.jl](https://julia.arviz.org/InferenceObjects): a base package implementing the [`InferenceData`](@ref) type with utilities for building, saving, and working with it
- [MCMCDiagnosticTools.jl](https://julia.arviz.org/MCMCDiagnosticTools): diagnostics for Markov Chain Monte Carlo methods
- [PSIS.jl](https://julia.arviz.org/PSIS): Pareto-smoothed importance sampling
- [PosteriorStats.jl](https://julia.arviz.org/PosteriorStats): common statistical analyses for the Bayesian workflow

Additional functionality can be loaded with the following packages:
- [ArviZExampleData.jl](https://julia.arviz.org/ArviZExampleData): example `InferenceData` objects, useful for demonstration and testing
- [ArviZPythonPlots.jl](https://julia.arviz.org/ArviZPythonPlots): Python ArviZ's library of plotting functions for Julia types

See the navigation bar for more useful packages.

## [Installation](@id installation)

From the Julia REPL, type `]` to enter the Pkg REPL mode and run

```
pkg> add ArviZ
```

## [Usage](@id usage)

See the [Quickstart](./quickstart) for example usage and the [API Overview](@ref api) for description of functions.

## [Extending ArviZ.jl](@id extendingarviz)

To use a custom data type with ArviZ.jl, simply overload [`InferenceObjects.convert_to_inference_data`](@ref) to convert your input(s) to an [`InferenceObjects.InferenceData`](@ref).
