using Conda

# temporary workaround for https://github.com/arviz-devs/ArviZ.jl/issues/188
Conda.add("scipy<=1.8.0")
Conda.add("matplotlib<3.6.0")
