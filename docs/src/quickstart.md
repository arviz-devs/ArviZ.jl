# [ArviZ.jl Quickstart](@id quickstart)

_This quickstart is adapted from [ArviZ's Quickstart](https://arviz-devs.github.io/arviz/notebooks/Introduction.html)._

```@setup quickstart
using PyPlot, ArviZ, Distributions, CmdStan, Pkg, InteractiveUtils

using PyCall
np = pyimport_conda("numpy", "numpy")
np.seterr(divide="ignore", invalid="ignore")

using Random
Random.seed!(42)

turing_chns = read("../src/assets/turing_centered_eight_chains.jls", MCMCChains.Chains)
```

```@example quickstart
using ArviZ

# ArviZ ships with style sheets!
ArviZ.use_style("arviz-darkgrid")
```

## Get started with plotting

ArviZ.jl is designed to be used with libraries like [CmdStan](https://github.com/StanJulia/CmdStan.jl) and [Turing](https://turing.ml) but works fine with raw arrays.

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

size = (10, 50)
plot_forest(Dict(
    "normal" => randn(size),
    "gumbel" => rand(Gumbel(), size),
    "student t" => rand(TDist(6), size),
    "exponential" => rand(Exponential(), size)
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

First we create our data.

```@example quickstart
J = 8
y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
sigma = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
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
nothing # hide
```

Now we write and run the model using Turing:

```julia
using Turing

@model centered_eight(J, y, sigma) = begin
    mu ~ Normal(0, 5)
    tau ~ Truncated(Cauchy(0, 5), 0, Inf)
    theta = tzeros(J)
    theta ~ [Normal(mu, tau)]
    y ~ MvNormal(theta, sigma)
end

nchains = 4
model = centered_eight(J, y, sigma)
sampler = NUTS(1000, 0.8)
turing_chns = mapreduce(chainscat, 1:nchains) do _
    return sample(model, sampler, 2000; progress = false)
end;
```

Most ArviZ functions work fine with `Chains` objects from Turing:

```@example quickstart
plot_autocorr(convert_to_inference_data(turing_chns); var_names=["mu", "tau"]);
savefig("quick_turingautocorr.svg"); nothing # hide
```

![](quick_turingautocorr.svg)

### Convert to `InferenceData`

For much more powerful querying, analysis and plotting, we can use built-in ArviZ utilities to convert `Chains` objects to xarray datasets.
Note we are also giving some information about labelling.

ArviZ is built to work with [`InferenceData`](@ref) (a netcdf datastore that loads data into `xarray` datasets), and the more *groups* it has access to, the more powerful analyses it can perform.
Here is a plot of the trace, which is common in Turing workflows.
Note the intelligent labels.

```@example quickstart
data = from_mcmcchains(
    turing_chns,
#     prior = prior, # hide
#     posterior_predictive = posterior_predictive, # hide
    library = "Turing",
    coords = Dict("school" => schools),
    dims = Dict("theta" => ["school"], "obs" => ["school"])
)
```

```@example quickstart
plot_trace(data);
savefig("quick_turingtrace.png"); nothing # hide
```

![](quick_turingtrace.png)

We can also generate summary stats

```@example quickstart
summarystats(data)
```

and examine the energy distribution of the Hamiltonian sampler

```@example quickstart
plot_energy(data);
savefig("quick_turingenergy.svg"); nothing # hide
```

![](quick_turingenergy.svg)

## Plotting with CmdStan.jl outputs

CmdStan.jl and StanSample.jl also default to producing `Chains` outputs, and we can easily plot these chains.

Here is the same centered eight schools model:

```@example quickstart
using CmdStan

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

schools_dat = Dict("J" => J, "y" => y, "sigma" => sigma)
stan_model = Stanmodel(
    model = schools_code,
    nchains = 4,
    num_warmup = 1000,
    num_samples = 1000,
    random = CmdStan.Random(8675309) # hide
)
_, stan_chns, _ = stan(stan_model, schools_dat, summary = false);
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
data = from_cmdstan(
    stan_chns;
    posterior_predictive = "y_hat",
#     observed_data = Dict("y" => schools_dat["y"]), # hide
    log_likelihood = "log_lik",
    coords = Dict("school" => schools),
    dims = Dict(
        "theta" => ["school"],
        "y" => ["school"],
        "log_lik" => ["school"],
        "y_hat" => ["school"],
        "theta_tilde" => ["school"]
    )
)
```

Here is a plot showing where the Hamiltonian sampler had divergences:

```@example quickstart
plot_pair(data; coords = Dict("school" => ["Choate", "Deerfield", "Phillips Andover"]), divergences = true);
savefig("quick_cmdstanpair.png"); nothing # hide
```

![](quick_cmdstanpair.png)
