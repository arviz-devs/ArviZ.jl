# [ArviZ.jl Quickstart](@id quickstart)

_This quickstart is adapted from [ArviZ's Quickstart](https://arviz-devs.github.io/arviz/notebooks/Introduction.html)._

```@setup quickstart
using Random
Random.seed!(42)

# use fancy HTML for xarray.Dataset if available
try
    ArviZ.xarray.set_options(display_style = "html")
catch
    nothing
end
```

```@example quickstart
using ArviZ

# ArviZ ships with style sheets!
ArviZ.use_style("arviz-darkgrid")
```

## Get started with plotting

ArviZ.jl is designed to be used with libraries like [CmdStan](https://github.com/StanJulia/CmdStan.jl), [Turing.jl](https://turing.ml), and [Soss.jl](https://github.com/cscherrer/Soss.jl) but works fine with raw arrays.

```@example quickstart
plot_posterior(randn(100_000));
savefig("quick_postarray.svg"); nothing # hide
```

![](quick_postarray.svg)

Plotting a dictionary of arrays, ArviZ.jl will interpret each key as the name of a different random variable.
Each row of an array is treated as an independent series of draws from the variable, called a _chain_.
Below, we have 10 chains of 50 draws each for four different distributions.

```@example quickstart
using Distributions

s = (10, 50)
plot_forest(Dict(
    "normal" => randn(s),
    "gumbel" => rand(Gumbel(), s),
    "student t" => rand(TDist(6), s),
    "exponential" => rand(Exponential(), s)
));
savefig("quick_forestdists.svg"); nothing # hide
```

![](quick_forestdists.svg)

## Plotting with Turing.jl

See the [notebook](https://nbviewer.jupyter.org/github/arviz-devs/ArviZ.jl/blob/master/examples/intro_turing.ipynb).

## Plotting with CmdStan.jl outputs

See the [notebook](https://nbviewer.jupyter.org/github/arviz-devs/ArviZ.jl/blob/master/examples/intro_stan.ipynb).

## Plotting with Soss.jl outputs

See the [notebook](https://nbviewer.jupyter.org/github/arviz-devs/ArviZ.jl/blob/master/examples/intro_soss.ipynb).
