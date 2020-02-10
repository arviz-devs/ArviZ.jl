# ## Plotting with Soss.jl outputs

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

# With Soss, we can define our model for the posterior and easily use it to draw samples from the prior, prior predictive, posterior, and posterior predictive distributions.
#
# First we define our model:

using Soss, NamedTupleTools

mod = Soss.@model (J, σ) begin
    μ ~ Normal(0, 5)
    τ ~ HalfCauchy(5)
    θ ~ Normal(μ, τ) |> iid(J)
    y ~ For(J) do j
        Normal(θ[j], σ[j])
    end
end

constant_data = (J = J, σ = σ)
param_mod = mod(; constant_data...)

# Then we draw from the prior and prior predictive distributions.

prior_prior_pred = map(1:nchains*nsamples) do _
    draw = rand(param_mod)
    return delete(draw, keys(constant_data))
end
prior = map(draw -> delete(draw, :y), prior_prior_pred)
prior_pred = map(draw -> delete(draw, (:μ, :τ, :θ)), prior_prior_pred);
#md nothing # hide

# Next, we draw from the posterior using [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl).

post = map(1:nchains) do _
    dynamicHMC(param_mod, (y = y,), nsamples)
end;
#md nothing # hide

# Finally, we use the posterior samples to draw from the posterior predictive distribution.

pred = predictive(mod, :μ, :τ, :θ)
post_pred = map(post) do post_draws
    map(post_draws) do post_draw
        pred_draw = rand(pred(post_draw)(constant_data))
        return delete(pred_draw, keys(constant_data))
    end
end;
#md nothing # hide

# Each Soss draw is a `NamedTuple`.
# Now we combine all of the samples to an `InferenceData`:

idata = from_namedtuple(
    post;
    posterior_predictive = post_pred,
    prior = [prior],
    prior_predictive = [prior_pred],
    observed_data = Dict("y" => y),
    constant_data = constant_data,
    coords = Dict("school" => schools),
    dims = Dict(
        "y" => ["school"],
        "σ" => ["school"],
        "θ" => ["school"],
    ),
    library = Soss,
)

# We can compare the prior and posterior predictive distributions:

plot_density(
    [idata.posterior_predictive, idata.prior_predictive];
    data_labels = ["Post-pred", "Prior-pred"],
    var_names = ["y"],
);
#md savefig("quick_sosspred.png"); nothing # hide
#md # ![](quick_sosspred.png)

# ## Environment

using Pkg
Pkg.status()
#-
versioninfo()
