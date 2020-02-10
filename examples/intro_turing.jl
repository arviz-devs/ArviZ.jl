# ## Plotting with MCMCChains.jl's `Chains` objects produced by Turing.jl

using Random
Random.seed!(42)

using ArviZ

## use fancy HTML for xarray.Dataset if available
#nb try
#nb     ArviZ.xarray.set_options(display_style = "html")
#nb catch
#nb     nothing
#nb end
#md try
#md     ArviZ.xarray.set_options(display_style = "html")
#md catch
#md     nothing
#md end

## ArviZ ships with style sheets!
ArviZ.use_style("arviz-darkgrid")

# ArviZ is designed to work well with high dimensional, labelled data.
# Consider the [eight schools model](https://statmodeling.stat.columbia.edu/2014/01/21/everything-need-know-bayesian-statistics-learned-eight-schools/), which roughly tries to measure the effectiveness of SAT classes at eight different schools.
# To show off ArviZ's labelling, I give the schools the names of [a different eight schools](https://en.wikipedia.org/wiki/Eight_Schools_Association).
#
# This model is small enough to write down, is hierarchical, and uses labelling.
# Additionally, a centered parameterization causes [divergences](https://mc-stan.org/users/documentation/case-studies/divergences_and_bias.html) (which are interesting for illustration).
#
# First we create our data and set some sampling parameters.

J = 8
y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
σ = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
schools = [
    "Choate",
    "Deerfield",
    "Phillips Andover",
    "Phillips Exeter",
    "Hotchkiss",
    "Lawrenceville",
    "St. Paul's",
    "Mt. Hermon"
]

nwarmup, nsamples, nchains = 1000, 1000, 4;
#md nothing # hide

# Now we write and run the model using Turing:

using Turing

Turing.@model turing_model(J, y, σ) = begin
    μ ~ Normal(0, 5)
    τ ~ truncated(Cauchy(0, 5), 0, Inf)
    θ ~ MvNormal(μ .* ones(J), τ .* ones(J))
    y ~ MvNormal(θ, σ)
end

param_mod = turing_model(J, y, σ)
sampler = NUTS(nwarmup, 0.8)
turing_chns = psample(
    param_mod,
    sampler,
    nwarmup + nsamples,
    nchains;
#md     progress = false,
#nb     progress = false,
);

# Most ArviZ functions work fine with `Chains` objects from Turing:

plot_autocorr(convert_to_inference_data(turing_chns); var_names = ["μ", "τ"]);
#md savefig("quick_turingautocorr.svg"); nothing # hide
#md # ![](quick_turingautocorr.svg)

# ### Convert to `InferenceData`
#
# For much more powerful querying, analysis and plotting, we can use built-in ArviZ utilities to convert `Chains` objects to xarray datasets.
# Note we are also giving some information about labelling.
#
# ArviZ is built to work with [`InferenceData`](@ref) (a netcdf datastore that loads data into `xarray` datasets), and the more *groups* it has access to, the more powerful analyses it can perform.

idata = from_mcmcchains(
    turing_chns,
    coords = Dict("school" => schools),
    dims = Dict(
        "y" => ["school"],
        "σ" => ["school"],
        "θ" => ["school"],
    ),
    library = "Turing",
)

# Each group is an [`ArviZ.Dataset`](@ref) (a thinly wrapped `xarray.Dataset`).
# We can view a summary of the dataset.

idata.posterior

# Here is a plot of the trace. Note the intelligent labels.

plot_trace(idata);
#md savefig("quick_turingtrace.png"); nothing # hide
#md # ![](quick_turingtrace.png)

# We can also generate summary stats

summarystats(idata)

# and examine the energy distribution of the Hamiltonian sampler

plot_energy(idata);
#md savefig("quick_turingenergy.svg"); nothing # hide
#md # ![](quick_turingenergy.svg)

# ## Environment

using Pkg
Pkg.status()
#-
versioninfo()
