# ArviZ

ArviZ.jl is a Julia wrapper for the Python package ArviZ for exploratory analysis of Bayesian models.

The reader is urged to consult ArviZ's documentation for features and usage information. This documentation will be limited to differences between the packages, applications using Julia's probabilistic programming languages (PPLs), and examples in Julia.

## Why ArviZ.jl

Besides removing the need to explicitly import ArviZ with PyCall, ArviZ.jl extends ArviZ with functionality for converting Julia types into ArviZ's `InferenceData` format. It also allows smoother usage with PyPlot.jl and Pandas.jl and provides functions that can be overloaded by other packages to enable their types to be used with ArviZ.

## Installation

To install ArviZ.jl with its Python dependencies in Julia's private conda environment, run

```console
PYTHON="" julia -e 'using Pkg; Pkg.add("https://github.com/sethaxen/ArviZ.jl")'
```

To use with the default Python environment, first install ArviZ. Then in Julia run

```julia
] add https://github.com/sethaxen/ArviZ.jl
```

## Design

ArviZ.jl wraps ArviZ's API functions and closely follows ArviZ's design. It also supports conversion of `MCMCChains`'s `Chains` as returned by Turing.jl, CmdStan.jl, StanSample.jl, and others into ArviZ's `InferenceData` format. See [Walkthrough]() for examples.

Issues and pull requests are welcome.
