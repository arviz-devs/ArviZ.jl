# [ArviZ.jl](@id arvizjl)

ArviZ.jl is a Julia interface to the [ArviZ](https://arviz-devs.github.io/arviz/) package for exploratory analysis of Bayesian models.

The reader is urged to consult [ArviZ's documentation](https://arviz-devs.github.io/arviz/) for detailed description of features and usage. This documentation will be limited to differences between the packages, applications using Julia's probabilistic programming languages (PPLs), and examples in Julia.

## [Purpose](@id purpose)

Besides removing the need to explicitly import ArviZ with [PyCall.jl](https://github.com/JuliaPy/PyCall.jl), ArviZ.jl extends ArviZ with functionality for converting Julia types into ArviZ's [`InferenceData`](https://arviz-devs.github.io/arviz/notebooks/XarrayforArviZ.html) format. It also allows smoother usage with [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl) and [Pandas.jl](https://github.com/JuliaPy/Pandas.jl) and provides functions that can be overloaded by other packages to enable their types to be used with ArviZ.

## [Installation](@id installation)

To install ArviZ.jl with its Python dependencies in Julia's private conda environment, in the console run

```console
PYTHON="" julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/sethaxen/ArviZ.jl"))'
```

To use with the default Python environment, first [install ArviZ](https://github.com/arviz-devs/arviz#installation). Then in Julia's REPL run

```julia
] add https://github.com/sethaxen/ArviZ.jl
```

## [Design](@id design)

ArviZ.jl supports all of ArviZ's [API](https://arviz-devs.github.io/arviz/api.html), except for its [Numba functionality](@ref knownissues). See ArviZ's API documentation for details.

ArviZ.jl wraps ArviZ's API functions and closely follows ArviZ's design. It also supports conversion of `MCMCChains`'s `Chains` as returned by [Turing.jl](https://turing.ml), [CmdStan.jl](https://github.com/StanJulia/CmdStan.jl), [StanSample.jl](https://github.com/StanJulia/StanSample.jl), and others into ArviZ's `InferenceData` format. See [Quickstart](@ref) for examples.

The package is intended to be used with [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl).

ArviZ.jl development occurs on [GitHub](https://github.com/sethaxen/ArviZ.jl). Issues and pull requests are welcome.

## [Differences from ArviZ](@id differences)

In ArviZ, functions in the [API](https://arviz-devs.github.io/arviz/api.html) are usually called with the package name prefix, (e.g. `arviz.plot_posterior`). In ArviZ.jl, most of the same functions are exported and therefore called without the prefix (e.g. `plot_posterior`). The exception are `from_xyz` converters for packages that have no (known) Julia wrappers. These functions are not exported to reduce namespace clutter.

ArviZ.jl transparently interconverts between `arviz.InferenceData` and our own `InferenceData`, used for dispatch. `InferenceData` has identical usage to its Python counterpart.

Functions that in ArviZ return Pandas types here return their [Pandas.jl](https://github.com/JuliaPy/Pandas.jl) wrappers, which are used the same way.

ArviZ includes the context managers `rc_context` and `interactive_backend`. ArviZ.jl includes functions that can be used with a nearly identical syntax. `interactive_backend` here is not limited to an IPython/IJulia context.

In place of `arviz.style.use` and `arviz.style.styles`, ArviZ.jl provides `ArviZ.use_style` and `ArviZ.styles`.

## [Known Issues](@id knownissues)

ArviZ.jl uses [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) to wrap ArviZ. At the moment, Julia segfaults if Numba is imported, which ArviZ does if it is available. For the moment, the workaround is to [specify a Python version](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version) that doesn't have Numba installed. See [this issue](https://github.com/JuliaPy/PyCall.jl/issues/220) for more details.
