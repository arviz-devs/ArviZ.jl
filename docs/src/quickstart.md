# [ArviZ.jl Quickstart](@id quickstart)

_This quickstart is adapted from [ArviZ's Quickstart](https://arviz-devs.github.io/arviz/notebooks/Introduction.html)._

```@setup quickstart
using PyPlot, ArviZ, Pkg
import MCMCChains

using PyCall
np = pyimport_conda("numpy", "numpy")
np.seterr(divide="ignore", invalid="ignore")

turing_chns = read("../src/assets/turing_centered_eight_chains.jls", MCMCChains.Chains)

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
using Random

rng = Random.MersenneTwister(42)
plot_posterior(randn(rng, 100_000));
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
    "normal" => randn(rng, s),
    "gumbel" => rand(rng, Gumbel(), s),
    "student t" => rand(rng, TDist(6), s),
    "exponential" => rand(rng, Exponential(), s)
));
savefig("quick_forestdists.svg"); nothing # hide
```

![](quick_forestdists.svg)

## Plotting with MCMCChains.jl's `Chains` objects produced by Turing.jl

ArviZ is designed to work well with high dimensional, labelled data.
Consider the [eight schools model](https://statmodeling.stat.columbia.edu/2014/01/21/everything-need-know-bayesian-statistics-learned-eight-schools/), which roughly tries to measure the effectiveness of SAT classes at eight different schools.
To show off ArviZ's labelling, I give the schools the names of [a different eight schools](https://en.wikipedia.org/wiki/Eight_Schools_Association).

This model is small enough to write down, is hierarchical, and uses labelling.
Additionally, a centered parameterization causes [divergences](https://mc-stan.org/users/documentation/case-studies/divergences_and_bias.html) (which are interesting for illustration).

First we create our data and set some sampling parameters.

```@example quickstart
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
];

nwarmup, nsamples, nchains = 1000, 1000, 4;
nothing # hide
```

Now we write and run the model using Turing:

```julia
using Turing

Turing.@model turing_model(
    J,
    y,
    σ,
    ::Type{TV} = Vector{Float64},
) where {TV} = begin
    μ ~ Normal(0, 5)
    τ ~ truncated(Cauchy(0, 5), 0, Inf)
    θ = TV(undef, J)
    θ .~ Normal(μ, τ)
    y ~ MvNormal(θ, σ)
end

param_mod = turing_model(J, y, σ)
sampler = NUTS(nwarmup, 0.8)

turing_chns = psample(
    param_mod,
    sampler,
    nwarmup + nsamples,
    nchains;
    progress = true,
);
```

Most ArviZ functions work fine with `Chains` objects from Turing:

```@example quickstart
plot_autocorr(convert_to_inference_data(turing_chns); var_names = ["μ", "τ"]);
savefig("quick_turingautocorr.svg"); nothing # hide
```

![](quick_turingautocorr.svg)

### Convert to `InferenceData`

For much more powerful querying, analysis and plotting, we can use built-in ArviZ utilities to convert `Chains` objects to xarray datasets.
Note we are also giving some information about labelling.

ArviZ is built to work with [`InferenceData`](@ref) (a netcdf datastore that loads data into `xarray` datasets), and the more *groups* it has access to, the more powerful analyses it can perform.

```@example quickstart
idata = from_mcmcchains(
    turing_chns,
#     prior = prior, # hide
#     posterior_predictive = posterior_predictive, # hide
    coords = Dict("school" => schools),
    dims = Dict(
        "y" => ["school"],
        "σ" => ["school"],
        "θ" => ["school"],
    ),
    library = "Turing",
)
```

Each group is an [`ArviZ.Dataset`](@ref) (a thinly wrapped `xarray.Dataset`).
We can view a summary of the dataset.

```@example quickstart
idata.posterior
```

Here is a plot of the trace. Note the intelligent labels.

```@example quickstart
plot_trace(idata);
savefig("quick_turingtrace.png"); nothing # hide
```

![](quick_turingtrace.png)

We can also generate summary stats

```@example quickstart
summarystats(idata)
```

and examine the energy distribution of the Hamiltonian sampler

```@example quickstart
plot_energy(idata);
savefig("quick_turingenergy.svg"); nothing # hide
```

![](quick_turingenergy.svg)

## Plotting with CmdStan.jl outputs

CmdStan.jl and StanSample.jl also default to producing `Chains` outputs, and we can easily plot these chains.

Here is the same centered eight schools model:

```@example quickstart
using CmdStan, MCMCChains

schools_code = """
data {
  int<lower=0> J;
  real y[J];
  real<lower=0> sigma[J];
}

parameters {
  real mu;
  real<lower=0> tau;
  real theta[J];
}

model {
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5);
  theta ~ normal(mu, tau);
  y ~ normal(theta, sigma);
}

generated quantities {
    vector[J] log_lik;
    vector[J] y_hat;
    for (j in 1:J) {
        log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
        y_hat[j] = normal_rng(theta[j], sigma[j]);
    }
}
"""

schools_dat = Dict("J" => J, "y" => y, "sigma" => σ)
stan_model = Stanmodel(
    model = schools_code,
    name = "schools",
    nchains = nchains,
    num_warmup = nwarmup,
    num_samples = nsamples,
    output_format = :mcmcchains,
    random = CmdStan.Random(8675309),
)
_, stan_chns, _ = stan(stan_model, schools_dat, summary = false);
Base.Filesystem.rm(stan_model.tmpdir; recursive = true, force = true); # hide
nothing # hide
```

```@example quickstart
plot_density(convert_to_inference_data(stan_chns); var_names=["mu", "tau"]);
savefig("quick_cmdstandensity.svg"); nothing # hide
```

![](quick_cmdstandensity.svg)

Again, converting to `InferenceData`, we can get much richer labelling and mixing of data.
Note that we're using the same [`from_cmdstan`](@ref) function used by ArviZ to process cmdstan output files, but through the power of dispatch in Julia, if we pass a `Chains` object, it instead uses ArviZ.jl's overloads, which forward to [`from_mcmcchains`](@ref).

```@example quickstart
idata = from_cmdstan(
    stan_chns;
    posterior_predictive = "y_hat",
    observed_data = Dict("y" => schools_dat["y"]),
    log_likelihood = "log_lik",
    coords = Dict("school" => schools),
    dims = Dict(
        "y" => ["school"],
        "sigma" => ["school"],
        "theta" => ["school"],
        "log_lik" => ["school"],
        "y_hat" => ["school"],
    ),
)
```

Here is a plot showing where the Hamiltonian sampler had divergences:

```@example quickstart
plot_pair(
    idata;
    coords = Dict("school" => ["Choate", "Deerfield", "Phillips Andover"]),
    divergences = true,
);
savefig("quick_cmdstanpair.png"); nothing # hide
```

![](quick_cmdstanpair.png)

## Plotting with Soss.jl outputs

With Soss, we can define our model for the posterior and easily use it to draw samples from the prior, prior predictive, posterior, and posterior predictive distributions.

First we define our model:

```@example quickstart
using Soss, NamedTupleTools

mod = Soss.@model (J, σ) begin
    μ ~ Normal(0, 5)
    τ ~ HalfCauchy(5)
    θ ~ Normal(μ, τ) |> iid(J)
    y ~ For(1:J) do j
        Normal(θ[j], σ[j])
    end
end

constant_data = (J = J, σ = σ)
param_mod = mod(; constant_data...)
```

Then we draw from the prior and prior predictive distributions.

```@example quickstart
Random.seed!(5298)
prior_prior_pred = map(1:nchains*nsamples) do _
    draw = rand(param_mod)
    return delete(draw, keys(constant_data))
end

prior = map(draw -> delete(draw, :y), prior_prior_pred)
prior_pred = map(draw -> delete(draw, (:μ, :τ, :θ)), prior_prior_pred);
nothing # hide
```

Next, we draw from the posterior using [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl).

```@example quickstart
post = map(1:nchains) do _
    dynamicHMC(param_mod, (y = y,), nsamples)
end;
nothing # hide
```

Finally, we use the posterior samples to draw from the posterior predictive distribution.

```@example quickstart
pred = predictive(mod, :μ, :τ, :θ)
post_pred = map(post) do post_draws
    map(post_draws) do post_draw
        pred_draw = rand(pred(post_draw)(constant_data))
        return delete(pred_draw, keys(constant_data))
    end
end;
nothing # hide
```

Each Soss draw is a `NamedTuple`.
Now we combine all of the samples to an `InferenceData`:

```@example quickstart
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
```

We can compare the prior and posterior predictive distributions:

```@example quickstart
plot_density(
    [idata.posterior_predictive, idata.prior_predictive];
    data_labels = ["Post-pred", "Prior-pred"],
    var_names = ["y"],
)
savefig("quick_sosspred.png"); nothing # hide
```

![](quick_sosspred.png)

## Environment

```@example quickstart
using Pkg
Pkg.status()
```

```@example quickstart
using InteractiveUtils
versioninfo()
```
