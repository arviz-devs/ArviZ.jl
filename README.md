[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Build Status](https://travis-ci.com/sethaxen/ArviZ.jl.svg?branch=master)](https://travis-ci.com/sethaxen/ArviZ.jl)
[![codecov.io](http://codecov.io/github/sethaxen/ArviZ.jl/coverage.svg?branch=master)](http://codecov.io/github/sethaxen/ArviZ.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://sethaxen.github.io/ArviZ.jl/dev)

# ArviZ.jl

ArviZ.jl is a Julia interface to the [ArviZ](https://arviz-devs.github.io/arviz/) package for exploratory analysis of Bayesian models.

See the [documentation](https://sethaxen.github.io/ArviZ.jl/dev) for details.

## Basic usage

This example uses a centered parameterization of
[ArviZ's version of the eight schools model](https://arviz-devs.github.io/arviz/notebooks/Introduction.html), which causes divergences, in [Turing.jl](https://turing.ml).

```julia
using ArviZ, PyPlot, Turing
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
