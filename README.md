# ArviZ.jl

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Build Status](https://travis-ci.org/sethaxen/ArviZ.jl.svg?branch=master)](https://travis-ci.org/sethaxen/ArviZ.jl)
[![codecov.io](http://codecov.io/github/sethaxen/ArviZ.jl/coverage.svg?branch=master)](http://codecov.io/github/sethaxen/ArviZ.jl?branch=master)

ArviZ.jl is a Julia interface to the
[ArviZ](https://arviz-devs.github.io/arviz/) package for exploratory analysis
of Bayesian models. It supports all of ArviZ's
[API](https://arviz-devs.github.io/arviz/api.html), except for its `Numba`
functionality.

This package also augments `ArviZ` to enable conversion from
[MCMCChains.jl](https://github.com/TuringLang/MCMCChains.jl)'s
`AbstractChains` type to `InferenceData`, which thinly wraps
`arviz.InferenceData` and is used the same way.

The package is meant to be used with
[PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl), whose API is exported.

## Installation

First [install ArviZ](https://github.com/arviz-devs/arviz#installation). Then,
within Julia, enter

```julia
] add https://github.com/sethaxen/ArviZ.jl
```

## Basic usage

This example uses a centered parameterization of
[ArviZ's version of the eight schools model](https://arviz-devs.github.io/arviz/notebooks/Introduction.html), which causes divergences, in [Turing.jl](https://turing.ml).

```julia
using ArviZ, Turing
ArviZ.use_style(["default", "arviz-darkgrid"])

# Turing model
@model school8(J, y, sigma) = begin
    mu ~ Normal(0, 5)
    tau ~ Truncated(Cauchy(0, 5), 0, Inf)
    theta ~ Normal(mu, tau)
    for j = 1:J
        y[j] ~ Normal(theta, sigma[j])
    end
end

J = 8
y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
sigma = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
model_fun = school8(J, y, sigma)
sampler = NUTS(0.8)
chn = mapreduce(c -> sample(model_fun, sampler, 1000), chainscat, 1:4) # 4 chains

data = convert_to_inference_data(chn) # convert MCMCChains.Chains to InferenceData
summary(data) # show summary statistics

plot_posterior(data) # plot posterior KDEs
display(gcf())

plot_pair(data; divergences=true) # plot pairwise scatter plot
display(gcf())
```

## Differences from ArviZ

In ArviZ, functions in the [API](https://arviz-devs.github.io/arviz/api.html)
are usually called with the package name prefix, (e.g. `arviz.plot_posterior`).
In ArviZ.jl, the same functions are called without the prefix
(e.g. `plot_posterior`).

ArviZ.jl transparently interconverts between `arviz.InferenceData` and
our own `InferenceData`, used for dispatch. `InferenceData` has identical usage
to its Python counterpart.

Functions that in ArviZ return Pandas types here return their
[Pandas.jl](https://github.com/JuliaPy/Pandas.jl) analogs, which are used the
same way.

ArviZ includes the context managers `rc_context` and `interactive_backend`.
ArviZ.jl includes functions that can be used with a nearly identical syntax.
`interactive_backend` here is not limited to an IPython context.

In place of `arviz.style.use` and `arviz.style.styles`, ArviZ.jl provides
`ArviZ.use_style` and `ArviZ.styles`.

## Known Issues

ArviZ.jl uses [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) to wrap ArviZ.
At the moment, Julia segfaults if Numba is imported, which ArviZ does if it is
available. For the moment, the workaround is to
[specify a Python version](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version)
that doesn't have Numba installed. See
[this issue](https://github.com/JuliaPy/PyCall.jl/issues/220) for more details.
