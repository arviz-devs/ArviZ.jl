# ArviZ.jl

ArviZ.jl is a Julia interface to the
[ArviZ](https://arviz-devs.github.io/arviz/) package for exploratory analysis
of Bayesian models. Most of ArviZ's plotting
[API](https://arviz-devs.github.io/arviz/api.html) is supported, with
diagnostics and more to come.

This package also augments `ArviZ` to enable conversion from
[MCMCChains.jl](https://github.com/TuringLang/MCMCChains.jl)'s
`AbstractChains` type to `InferenceData`, which thinly wraps
`arviz.InferenceData` and is used the same way.

The package is meant to be used with
[PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl), whose api is exported.

## Differences from ArviZ

In place of `arviz.style.use`, ArviZ.jl provides `ArviZ.use_style`.

ArviZ.jl thinly wraps `arviz.InferenceData` in `InferenceData` for dispatch.

Functions in ArviZ that return Pandas types here return their
[Pandas.jl](https://github.com/JuliaPy/Pandas.jl) analogs.

## Basic usage

This example uses a centered parameterization of
[ArviZ's version of the eight schools model](https://arviz-devs.github.io/arviz/notebooks/Introduction.html), which causes divergences, in [Turing.jl](https://turing.ml).

```julia
using ArviZ
ArviZ.use_style("arviz-darkgrid")

using Turing
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
chn = mapreduce(c -> sample(model_fun, sampler, 1000), chainscat, 1:4)
data = convert_to_inference_data(chn)
summary(data) # summary statistics
plot_posterior(data) # posterior KDEs
plot_pair(data; divergences=true) # pairwise scatter plot
```

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/sdaxen/ArviZ.jl.svg?branch=master)](https://travis-ci.com/sdaxen/ArviZ.jl)
[![codecov.io](http://codecov.io/github/sdaxen/ArviZ.jl/coverage.svg?branch=master)](http://codecov.io/github/sdaxen/ArviZ.jl?branch=master)
