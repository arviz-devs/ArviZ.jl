# [ArviZ.jl: Exploratory analysis of Bayesian models in Julia](@id arvizjl)

![CI](https://github.com/arviz-devs/ArviZ.jl/workflows/CI/badge.svg)
[![codecov.io](https://codecov.io/github/arviz-devs/ArviZ.jl/coverage.svg?branch=main)](https://codecov.io/github/arviz-devs/ArviZ.jl?branch=main)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

ArviZ.jl is a Julia package for exploratory analysis of Bayesian models.
It is codeveloped with the Python package [ArviZ](https://python.arviz.org/) and in some cases temporarily relies on the Python package for functionality.

## [Installation](@id installation)

To install ArviZ.jl, we first need to install Python ArviZ.
To use with the default Python environment, first [install Python ArviZ](https://python.arviz.org/en/latest/getting_started/Installation.html).
From the Julia REPL, type `]` to enter the Pkg REPL mode and run

```
pkg> add ArviZ
```

To install ArviZ.jl with its Python dependencies in Julia's private conda environment, in the console run

```console
PYTHON="" julia -e 'using Pkg; Pkg.add("PyCall"); Pkg.build("PyCall"); Pkg.add("ArviZ")'
```

For specifying other Python versions, see the [PyCall documentation](https://github.com/JuliaPy/PyCall.jl).

## [Usage](@id usage)

See the [Quickstart](@ref) for example usage and the [API](@ref) for description of functions.

## [Extending ArviZ.jl](@id extendingarviz)

To use a custom data type with ArviZ.jl, simply overload [`convert_to_inference_data`](@ref) to convert your input(s) to an [`InferenceData`](@ref).

## [Known Issues](@id knownissues)

ArviZ.jl uses [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) to wrap ArviZ.
At the moment, Julia segfaults if Numba is imported, which ArviZ does if it is available.
For the moment, the workaround is to [specify a Python version](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version) that doesn't have Numba installed.
See [this issue](https://github.com/JuliaPy/PyCall.jl/issues/220) for more details.
