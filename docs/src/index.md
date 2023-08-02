# [ArviZ.jl: Exploratory analysis of Bayesian models in Julia](@id arvizjl)

ArviZ.jl is a Julia package for exploratory analysis of Bayesian models.
It is part of the [ArviZ project](https://www.arviz.org/), which also includes a related [Python package](https://python.arviz.org/).

## [Installation](@id installation)

From the Julia REPL, type `]` to enter the Pkg REPL mode and run

```
pkg> add ArviZ
```

## [Usage](@id usage)

See the [Quickstart](./quickstart) for example usage and the [API Overview](@ref api) for description of functions.

## [Extending ArviZ.jl](@id extendingarviz)

To use a custom data type with ArviZ.jl, simply overload [`convert_to_inference_data`](@ref) to convert your input(s) to an [`InferenceData`](@ref).
