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
    library=Turing,
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

We would like to compute additional useful groups to add to the [`InferenceData`](@ref).
ArviZ includes a Turing-specific converter [`from_turing`](@ref) that, given a model, posterior samples, and data, can add the missing groups:

```@example turing
idata = from_turing(
    turing_chns;
    model=param_mod,
    observed_data=Dict("y" => y),
    rng=rng,
    coords=Dict("school" => schools),
    dims=Dict("y" => ["school"], "σ" => ["school"], "θ" => ["school"]),
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
using Soss, NamedTupleTools

mod = Soss.@model (J, σ) begin
    μ ~ Normal(0, 5)
    τ ~ HalfCauchy(5)
    θ ~ iid(J)(Normal(μ, τ))
    y ~ For(1:J) do j
        Normal(θ[j], σ[j])
    end
end

constant_data = (J=J, σ=σ)
param_mod = mod(; constant_data...)
```

Then we draw from the prior and prior predictive distributions.

```@example soss
Random.seed!(5298)
prior_priorpred = [
    map(1:(nchains * nsamples)) do _
        draw = rand(param_mod)
        return delete(draw, keys(constant_data))
    end,
];
nothing # hide
```

Next, we draw from the posterior using [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl).

```@example soss
post = map(1:nchains) do _
    dynamicHMC(param_mod, (y=y,), nsamples)
end;
nothing # hide
```

Finally, we update the posterior samples with draws from the posterior predictive distribution.

```@example soss
pred = predictive(mod, :μ, :τ, :θ)
post_postpred = map(post) do post_draws
    map(post_draws) do post_draw
        pred_draw = rand(pred(post_draw))
        pred_draw = delete(pred_draw, keys(constant_data))
        return merge(pred_draw, post_draw)
    end
end;
nothing # hide
```

Each Soss draw is a `NamedTuple`.
We can plot the rank order statistics of the posterior to identify poor convergence:

```@example soss
plot_rank(post; var_names=["μ", "τ"]);
gcf()
```

Now we combine all of the samples to an `InferenceData`:

```@example soss
idata = from_namedtuple(
    post_postpred;
    posterior_predictive=[:y],
    prior=prior_priorpred,
    prior_predictive=[:y],
    observed_data=(y=y,),
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
Text(sprint(io -> Pkg.status(io=io)))
```

```@example
using InteractiveUtils
versioninfo()
```
