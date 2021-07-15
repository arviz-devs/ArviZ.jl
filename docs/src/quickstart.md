# [ArviZ.jl Quickstart](@id quickstart)

!!! note
    
    This tutorial is adapted from [ArviZ's quickstart](https://arviz-devs.github.io/arviz/getting_started/Introduction.html).

```@example gettingstarted
using ArviZ
using PyPlot

# ArviZ ships with style sheets!
ArviZ.use_style("arviz-darkgrid")
```

## Get started with plotting

ArviZ.jl is designed to be used with libraries like [CmdStan](https://github.com/StanJulia/CmdStan.jl), [Turing.jl](https://turing.ml), and [Soss.jl](https://github.com/cscherrer/Soss.jl) but works fine with raw arrays.

```@example gettingstarted
using Random

rng = Random.MersenneTwister(37772)
plot_posterior(randn(rng, 100_000));
gcf()
```

Plotting a dictionary of arrays, ArviZ.jl will interpret each key as the name of a different random variable.
Each row of an array is treated as an independent series of draws from the variable, called a _chain_.
Below, we have 10 chains of 50 draws each for four different distributions.

```@example gettingstarted
using Distributions

s = (10, 50)
plot_forest(
    Dict(
        "normal" => randn(rng, s),
        "gumbel" => rand(rng, Gumbel(), s),
        "student t" => rand(rng, TDist(6), s),
        "exponential" => rand(rng, Exponential(), s),
    ),
);
gcf()
```

## Plotting with MCMCChains.jl's `Chains` objects produced by Turing.jl

```@setup turing
using PyPlot, Random, ArviZ 
ArviZ.use_style("arviz-darkgrid")
```

ArviZ is designed to work well with high dimensional, labelled data.
Consider the [eight schools model](https://statmodeling.stat.columbia.edu/2014/01/21/everything-need-know-bayesian-statistics-learned-eight-schools/), which roughly tries to measure the effectiveness of SAT classes at eight different schools.
To show off ArviZ's labelling, I give the schools the names of [a different eight schools](https://en.wikipedia.org/wiki/Eight_Schools_Association).

This model is small enough to write down, is hierarchical, and uses labelling.
Additionally, a centered parameterization causes [divergences](https://mc-stan.org/users/documentation/case-studies/divergences_and_bias.html) (which are interesting for illustration).

First we create our data and set some sampling parameters.

```@example turing
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
    "Mt. Hermon",
];

nwarmup, nsamples, nchains = 1000, 1000, 4;
nothing # hide
```

Now we write and run the model using Turing:

```@example turing
using Turing

Turing.@model function turing_model(y, σ, J=length(y))
    μ ~ Normal(0, 5)
    τ ~ truncated(Cauchy(0, 5), 0, Inf)
    θ ~ filldist(Normal(μ, τ), J)
    for i in 1:J
        y[i] ~ Normal(θ[i], σ[i])
    end
end

param_mod = turing_model(y, σ)
sampler = NUTS(nwarmup, 0.8)

rng = Random.MersenneTwister(16653)
turing_chns = sample(
    rng, param_mod, sampler, MCMCThreads(), nsamples, nchains; progress=false
);
nothing # hide
```

Most ArviZ functions work fine with `Chains` objects from Turing:

```@example turing
plot_autocorr(turing_chns; var_names=["μ", "τ"]);
gcf()
```

### Convert to `InferenceData`

For much more powerful querying, analysis and plotting, we can use built-in ArviZ utilities to convert `Chains` objects to xarray datasets.
Note we are also giving some information about labelling.

ArviZ is built to work with [`InferenceData`](@ref) (a netcdf datastore that loads data into `xarray` datasets), and the more *groups* it has access to, the more powerful analyses it can perform.

```@example turing
idata = from_mcmcchains(
    turing_chns;
    coords=Dict("school" => schools),
    dims=Dict("y" => ["school"], "σ" => ["school"], "θ" => ["school"]),
    library="Turing",
)
```

Each group is an [`ArviZ.Dataset`](@ref) (a thinly wrapped `xarray.Dataset`).
We can view a summary of the dataset.

```@example turing
idata.posterior
```

Here is a plot of the trace. Note the intelligent labels.

```@example turing
plot_trace(idata);
gcf()
```

We can also generate summary stats

```@example turing
summarystats(idata)
```

and examine the energy distribution of the Hamiltonian sampler

```@example turing
plot_energy(idata);
gcf()
```

### Additional information in Turing.jl

With a few more steps, we can use Turing to compute additional useful groups to add to the [`InferenceData`](@ref).

To sample from the prior, one simply calls `sample` but with the `Prior` sampler:

```@example turing
prior = sample(param_mod, Prior(), nsamples; progress=false)
```

To draw from the prior and posterior predictive distributions we can instantiate a "predictive model", i.e. a Turing model but with the observations set to `missing`, and then calling `predict` on the predictive model and the previously drawn samples:

```@example turing
# Instantiate the predictive model
param_mod_predict = turing_model(similar(y, Missing), σ)
# and then sample!
prior_predictive = predict(param_mod_predict, prior)
posterior_predictive = predict(param_mod_predict, turing_chns)
```

And to extract the pointwise log-likelihoods, which is useful if you want to compute metrics such as [`loo`](@ref),

```@example turing
loglikelihoods = Turing.pointwise_loglikelihoods(
    param_mod, MCMCChains.get_sections(turing_chns, :parameters)
)
```

This can then be included in the [`from_mcmcchains`](@ref) call from above:

```@example turing
using LinearAlgebra
# Ensure the ordering of the loglikelihoods matches the ordering of `posterior_predictive`
ynames = string.(keys(posterior_predictive))
loglikelihoods_vals = getindex.(Ref(loglikelihoods), ynames)
# Reshape into `(nchains, nsamples, size(y)...)`
loglikelihoods_arr = permutedims(cat(loglikelihoods_vals...; dims=3), (2, 1, 3))

idata = from_mcmcchains(
    turing_chns;
    posterior_predictive=posterior_predictive,
    log_likelihood=Dict("y" => loglikelihoods_arr),
    prior=prior,
    prior_predictive=prior_predictive,
    observed_data=Dict("y" => y),
    coords=Dict("school" => schools),
    dims=Dict("y" => ["school"], "σ" => ["school"], "θ" => ["school"]),
    library="Turing",
)
```

Then we can for example compute the expected *leave-one-out (LOO)* predictive density, which is an estimate of the out-of-distribution predictive fit of the model:

```@example turing
loo(idata) # higher is better
```

If the model is well-calibrated, i.e. it replicates the true generative process well, the CDF of the pointwise LOO values should be similarly distributed to a uniform distribution.
This can be inspected visually:

```@example turing
plot_loo_pit(idata; y="y", ecdf=true);
gcf()
```

## Plotting with CmdStan.jl outputs

```@setup cmdstan
using PyPlot, ArviZ 
ArviZ.use_style("arviz-darkgrid")

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
    "Mt. Hermon",
];
nwarmup, nsamples, nchains = 1000, 1000, 4
```

CmdStan.jl and StanSample.jl also default to producing `Chains` outputs, and we can easily plot these chains.

Here is the same centered eight schools model:

```@example cmdstan
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
stan_model = Stanmodel(;
    model=schools_code,
    name="schools",
    nchains=nchains,
    num_warmup=nwarmup,
    num_samples=nsamples,
    output_format=:mcmcchains,
    random=CmdStan.Random(28983),
)
_, stan_chns, _ = stan(stan_model, schools_dat; summary=false);
Base.Filesystem.rm(stan_model.tmpdir; recursive=true, force=true); # hide
nothing # hide
```

```@example cmdstan
plot_density(stan_chns; var_names=["mu", "tau"]);
gcf()
```

Again, converting to `InferenceData`, we can get much richer labelling and mixing of data.
Note that we're using the same [`from_cmdstan`](@ref) function used by ArviZ to process cmdstan output files, but through the power of dispatch in Julia, if we pass a `Chains` object, it instead uses ArviZ.jl's overloads, which forward to [`from_mcmcchains`](@ref).

```@example cmdstan
idata = from_cmdstan(
    stan_chns;
    posterior_predictive="y_hat",
    observed_data=Dict("y" => schools_dat["y"]),
    log_likelihood="log_lik",
    coords=Dict("school" => schools),
    dims=Dict(
        "y" => ["school"],
        "sigma" => ["school"],
        "theta" => ["school"],
        "log_lik" => ["school"],
        "y_hat" => ["school"],
    ),
)
```

Here is a plot showing where the Hamiltonian sampler had divergences:

```@example cmdstan
plot_pair(
    idata;
    coords=Dict("school" => ["Choate", "Deerfield", "Phillips Andover"]),
    divergences=true,
);
gcf()
```

## Plotting with Soss.jl outputs

```@setup soss
using PyPlot, ArviZ, Random
ArviZ.use_style("arviz-darkgrid")

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
    "Mt. Hermon",
];
nwarmup, nsamples, nchains = 1000, 1000, 4
```

With Soss, we can define our model for the posterior and easily use it to draw samples from the prior, prior predictive, posterior, and posterior predictive distributions.

First we define our model:

```@example soss
using Soss
using Soss.MeasureTheory: HalfCauchy, Normal

mod = Soss.@model (J, σ) begin
    μ ~ Normal(μ=0, σ=5)
    τ ~ HalfCauchy(σ=5)
    θ ~ Normal(μ=μ, σ=τ) |> iid(J)
    y ~ For(1:J) do j
        Normal(μ=θ[j], σ=σ[j])
    end
end

constant_data = (J=J, σ=σ)
observed_data = (y=y,)
param_mod = mod(; constant_data...)
```

Then we draw from the prior and prior predictive distributions.

```@example soss
rng = Random.seed!(5298)
prior_priorpred = map(_ -> rand(rng, param_mod, nsamples), 1:nchains);
nothing # hide
```

Soss returns predictive samples in a `TupleVector`, which is an efficient way of storing a vector of `NamedTuple`s.
Next, we draw from the posterior using [SampleChainsDynamicHMC.jl](https://github.com/cscherrer/SampleChainsDynamicHMC.jl).

```@example soss
using SampleChainsDynamicHMC: dynamichmc
post = Soss.sample(rng, param_mod | observed_data, dynamichmc(), nsamples, nchains)
```

Soss returns posterior samples in a `SampleChains.AbstractChain` or, for multiple chains, in a `SampleChains.MultiChain`.
Finally, we update the posterior samples with draws from the posterior predictive distribution.

```@example soss
mod_pred = Soss.predictive(mod, :μ, :τ, :θ)
postpred = map(getchains(post)) do chain
    map(draw -> rand(rng, mod_pred(; constant_data..., draw...)), chain)
end;
nothing # hide
```

We can plot the rank order statistics of the posterior to identify poor convergence:

```@example soss
plot_rank(post; var_names=["μ", "τ"]);
gcf()
```

Now we combine all of the samples to an `InferenceData`:

```@example soss
idata = from_samplechains(
    post;
    posterior_predictive=postpred,
    prior=prior_priorpred,
    prior_predictive=[:y],
    observed_data=observed_data,
    constant_data=constant_data,
    coords=Dict("school" => schools),
    dims=Dict("y" => ["school"], "σ" => ["school"], "θ" => ["school"]),
    library=Soss,
)
```

We can compare the prior and posterior predictive distributions:

```@example soss
plot_density(
    [idata.posterior_predictive, idata.prior_predictive];
    data_labels=["Post-pred", "Prior-pred"],
    var_names=["y"],
)
gcf()
```

## Environment

```@example
using Pkg
Pkg.status()
```

```@example
using InteractiveUtils
versioninfo()
```
