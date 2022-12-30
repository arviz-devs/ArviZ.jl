using Conda

# temporary workaround for
# - https://github.com/arviz-devs/ArviZ.jl/issues/188
# - https://github.com/arviz-devs/arviz/issues/2120
Conda.add(["matplotlib<3.6.0", "scipy<=1.8.0"])

