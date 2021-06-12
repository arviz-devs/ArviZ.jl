### A Pluto.jl notebook ###
# v0.15.0

using Markdown
using InteractiveUtils

# ╔═╡ 4d0e37f3-85f5-4ad8-bbba-8e5a3288f48b
begin
    using PlutoUI
    PlutoUI.TableOfContents()
end

# ╔═╡ 467c2d13-6bfe-4feb-9626-fb14796168aa
begin
    using ArviZ, CmdStan, Distributions, LinearAlgebra, MCMCChains
    using NamedTupleTools, PyPlot, Random, Soss, Turing
end

# ╔═╡ d2eedd48-48c6-4fcd-b179-6be7fe68d3d6
md"""
## Setup

Here we add the necessary packages for this notebook and load a few we will use throughout.
"""

# ╔═╡ 06b00794-e97f-472b-b526-efe4815103f8
# ArviZ ships with style sheets!
ArviZ.use_style("arviz-darkgrid")

# ╔═╡ 5acbfd9a-cfea-4c3c-a3f0-1744eb7e4e27
md"""
## Get started with plotting

ArviZ.jl is designed to be used with libraries like [CmdStan](https://github.com/StanJulia/CmdStan.jl), [Turing.jl](https://turing.ml), and [Soss.jl](https://github.com/cscherrer/Soss.jl) but works fine with raw arrays.
"""

# ╔═╡ efb3f0af-9fac-48d8-bbb2-2dd6ebd5e4f6
rng1 = Random.MersenneTwister(37772)

# ╔═╡ 401e9b91-0bca-4369-8d36-3d9f0b3ad60b
begin
    plot_posterior(randn(rng1, 100_000))
    gcf()
end

# ╔═╡ 2c718ea5-2800-4df6-b62c-e0a9e440a1c3
md"""
Plotting a dictionary of arrays, ArviZ.jl will interpret each key as the name of a different random variable.
Each row of an array is treated as an independent series of draws from the variable, called a _chain_.
Below, we have 10 chains of 50 draws each for four different distributions.
"""

# ╔═╡ 49f19c17-ac1d-46b5-a655-4376b7713244
begin
    s = (10, 50)
    plot_forest(
        Dict(
            "normal" => randn(rng1, s),
            "gumbel" => rand(rng1, Gumbel(), s),
            "student t" => rand(rng1, TDist(6), s),
            "exponential" => rand(rng1, Exponential(), s),
        ),
    )
    gcf()
end

# ╔═╡ a9789109-2b90-40f7-926c-e7c87025d15f
md"""
## Plotting with MCMCChains.jl's `Chains` objects produced by Turing.jl

ArviZ is designed to work well with high dimensional, labelled data.
Consider the [eight schools model](https://statmodeling.stat.columbia.edu/2014/01/21/everything-need-know-bayesian-statistics-learned-eight-schools/), which roughly tries to measure the effectiveness of SAT classes at eight different schools.
To show off ArviZ's labelling, I give the schools the names of [a different eight schools](https://en.wikipedia.org/wiki/Eight_Schools_Association).

This model is small enough to write down, is hierarchical, and uses labelling.
Additionally, a centered parameterization causes [divergences](https://mc-stan.org/users/documentation/case-studies/divergences_and_bias.html) (which are interesting for illustration).

First we create our data and set some sampling parameters.
"""

# ╔═╡ 69124a94-a6aa-43f6-8d4f-fa9a6b1cfef1
begin
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
    ]
    ndraws = 1_000
    ndraws_warmup = 1_000
    nchains = 4
end;

# ╔═╡ 10986e0e-55f7-40ea-ab63-274fbab3126d
md"""
Now we write and run the model using Turing:
"""

# ╔═╡ f383d541-e22d-44b4-b8cb-28b3d67944a1
Turing.@model function model_turing(y, σ, J=length(y))
    μ ~ Normal(0, 5)
    τ ~ truncated(Cauchy(0, 5), 0, Inf)
    θ ~ filldist(Normal(μ, τ), J)
    for i in 1:J
        y[i] ~ Normal(θ[i], σ[i])
    end
end

# ╔═╡ 85bbcba7-c0f9-4c86-9cdf-a27055d3d448
begin
    param_mod_turing = model_turing(y, σ)
    sampler = NUTS(ndraws_warmup, 0.8)

    rng2 = Random.MersenneTwister(16653)
    turing_chns = sample(rng2, param_mod_turing, sampler, MCMCThreads(), ndraws, nchains)
end;

# ╔═╡ bd4ab044-51ce-4af9-83b2-bd8fc827f810
md"""
Most ArviZ functions work fine with `Chains` objects from Turing:
"""

# ╔═╡ 500f4e0d-0a36-4b5c-8900-667560fbf1d4
begin
    plot_autocorr(turing_chns; var_names=["μ", "τ"])
    gcf()
end

# ╔═╡ 1129ad94-f65a-4332-b354-21bcf7e53541
md"""
### Convert to `InferenceData`

For much more powerful querying, analysis and plotting, we can use built-in ArviZ utilities to convert `Chains` objects to xarray datasets.
Note we are also giving some information about labelling.

ArviZ is built to work with [`InferenceData`](@ref) (a netcdf datastore that loads data into `xarray` datasets), and the more *groups* it has access to, the more powerful analyses it can perform.
"""

# ╔═╡ 803efdd8-656e-4e37-ba36-81195d064972
idata_turing_post = from_mcmcchains(
    turing_chns;
    coords=Dict("school" => schools),
    dims=Dict("y" => ["school"], "σ" => ["school"], "θ" => ["school"]),
    library="Turing",
)

# ╔═╡ 79f342c8-0738-432b-bfd7-2da25e50fa91
md"""
Each group is an [`ArviZ.Dataset`](@ref) (a thinly wrapped `xarray.Dataset`).
We can view a summary of the dataset.
"""

# ╔═╡ 6209f947-5001-4507-b3e8-9747256f3328
idata_turing_post.posterior

# ╔═╡ 26692722-db0d-41de-b58c-339b639f6948
md"""
Here is a plot of the trace. Note the intelligent labels.
"""

# ╔═╡ 14046b83-9c0a-4d33-ae4e-36c7d6f1b2e6
begin
    plot_trace(idata_turing_post)
    gcf()
end

# ╔═╡ 737f319c-1ddd-45f2-8d10-aaecdc1334be
md"We can also generate summary stats..."

# ╔═╡ 5be3131d-db8b-44c8-9fcc-571d68695148
summarystats(idata_turing_post)

# ╔═╡ 0787a842-63ca-407d-9d07-a3d949940f92
md"...and examine the energy distribution of the Hamiltonian sampler."

# ╔═╡ 6e8343c8-bee3-4e1d-82d6-1885bfd1dbec
begin
    plot_energy(idata_turing_post)
    gcf()
end

# ╔═╡ cba6f6c9-82c4-4957-acc3-36e9f1c95d76
md"""
### Additional information in Turing.jl

With a few more steps, we can use Turing to compute additional useful groups to add to the [`InferenceData`](@ref).

To sample from the prior, one simply calls `sample` but with the `Prior` sampler:
"""

# ╔═╡ 79374a8b-98c1-443c-acdd-0ee00bd42b38
prior = sample(rng2, param_mod_turing, Prior(), ndraws);

# ╔═╡ c23562ba-3b0b-49d9-bb41-2fedaa9e9500
md"""
To draw from the prior and posterior predictive distributions we can instantiate a "predictive model", i.e. a Turing model but with the observations set to `missing`, and then calling `predict` on the predictive model and the previously drawn samples:
"""

# ╔═╡ d0b447fe-26b2-48e0-bcec-d9f04697973a
begin
    # Instantiate the predictive model
    param_mod_predict = model_turing(similar(y, Missing), σ)
    # and then sample!
    prior_predictive = Turing.predict(rng2, param_mod_predict, prior)
    posterior_predictive = Turing.predict(rng2, param_mod_predict, turing_chns)
end;

# ╔═╡ 4d2fdcbe-c1d4-43b6-b382-f2e956b952a1
md"""
And to extract the pointwise log-likelihoods, which is useful if you want to compute metrics such as [`loo`](@ref),
"""

# ╔═╡ 5a075722-232f-40fc-a499-8dc5b0c2424a
begin
    loglikelihoods = Turing.pointwise_loglikelihoods(
        param_mod_turing, MCMCChains.get_sections(turing_chns, :parameters)
    )
    # Ensure the ordering of the loglikelihoods matches the ordering of `posterior_predictive`
    ynames = string.(keys(posterior_predictive))
    loglikelihoods_vals = getindex.(Ref(loglikelihoods), ynames)
    # Reshape into `(nchains, nsamples, size(y)...)`
    loglikelihoods_arr = permutedims(cat(loglikelihoods_vals...; dims=3), (2, 1, 3))
end;

# ╔═╡ 1b5af2c3-f2ce-4e9d-9ad7-ac287a9178e2
md"This can then be included in the [`from_mcmcchains`](@ref) call from above:"

# ╔═╡ b38c7a43-f00c-43c0-aa6b-9c581d6d0c73
idata_turing = from_mcmcchains(
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

# ╔═╡ a3b71e4e-ac1f-4404-b8a2-35d03d326774
md"""
Then we can for example compute the expected *leave-one-out (LOO)* predictive density, which is an estimate of the out-of-distribution predictive fit of the model:
"""

# ╔═╡ f552b5b5-9744-41df-af90-46405367ea0b
loo(idata_turing) # higher is better

# ╔═╡ 9d3673f5-b57b-432e-944a-70b23643128a
md"""
If the model is well-calibrated, i.e. it replicates the true generative process well, the CDF of the pointwise LOO values should be similarly distributed to a uniform distribution.
This can be inspected visually:
"""

# ╔═╡ 05c9be29-7758-4324-971c-5579f99aaf9d
begin
    plot_loo_pit(idata_turing; y="y", ecdf=true)
    gcf()
end

# ╔═╡ 98acc304-22e3-4e6b-a2f4-d22f6847145b
md"""
## Plotting with CmdStan.jl outputs

CmdStan.jl and StanSample.jl also default to producing `Chains` outputs, and we can easily plot these chains.

Here is the same centered eight schools model:
"""

# ╔═╡ b46af168-1ce3-4058-a014-b66c645a6e0d
begin
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

    schools_data = Dict("J" => J, "y" => y, "sigma" => σ)
    stan_chns = mktempdir() do path
        stan_model = Stanmodel(;
            model=schools_code,
            name="schools",
            nchains=nchains,
            num_warmup=ndraws_warmup,
            num_samples=ndraws,
            output_format=:mcmcchains,
            random=CmdStan.Random(28983),
            tmpdir=path,
        )
        _, chns, _ = stan(stan_model, schools_data; summary=false)
        return chns
    end
end;

# ╔═╡ ab145e41-b230-4cad-bef5-f31e0e0770d4
begin
    plot_density(stan_chns; var_names=["mu", "tau"])
    gcf()
end

# ╔═╡ ffc7730c-d861-48e8-b173-b03e0542f32b
md"""
Again, converting to `InferenceData`, we can get much richer labelling and mixing of data.
Note that we're using the same [`from_cmdstan`](@ref) function used by ArviZ to process cmdstan output files, but through the power of dispatch in Julia, if we pass a `Chains` object, it instead uses ArviZ.jl's overloads, which forward to [`from_mcmcchains`](@ref).
"""

# ╔═╡ 020cbdc0-a0a2-4d20-838f-c99b541d5832
idata_stan = from_cmdstan(
    stan_chns;
    posterior_predictive="y_hat",
    observed_data=Dict("y" => schools_data["y"]),
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

# ╔═╡ e44b260c-9d2f-43f8-a64b-04245a0a5658
md"""Here is a plot showing where the Hamiltonian sampler had divergences:"""

# ╔═╡ 5070bbbc-68d2-49b8-bd91-456dc0da4573
begin
    plot_pair(
        idata_stan;
        coords=Dict("school" => ["Choate", "Deerfield", "Phillips Andover"]),
        divergences=true,
    )
    gcf()
end

# ╔═╡ 2674d532-a337-471e-8ba7-02b430f49f12
md"""
## Plotting with Soss.jl outputs

With Soss, we can define our model for the posterior and easily use it to draw samples from the prior, prior predictive, posterior, and posterior predictive distributions.

First we define our model:
"""

# ╔═╡ 14408abe-a16f-4cc0-a6f3-0bb2645653b7
constant_data = (J=J, σ=σ)

# ╔═╡ 9daec35c-3d6e-443c-87f9-213d51964f75
model_soss = Soss.@model (J, σ) begin
    μ ~ Normal(0, 5)
    τ ~ HalfCauchy(5)
    θ ~ iid(J)(Normal(μ, τ))
    y ~ For(1:J) do j
        Normal(θ[j], σ[j])
    end
end

# ╔═╡ 6a78e4a8-86c4-4438-b9bb-7c433d2bc8c8
param_mod_soss = model_soss(; constant_data...)

# ╔═╡ a94b3ad6-2305-43f6-881f-d871286b906a
md"Then we draw from the prior and prior predictive distributions."

# ╔═╡ d385ceea-beb5-4ec7-b50d-be691266440b
begin
    rng3 = MersenneTwister(5298)
    prior_priorpred = [
        map(1:(nchains * ndraws)) do _
            draw = rand(rng2, param_mod_soss)
            return delete(draw, keys(constant_data))
        end,
    ]
end

# ╔═╡ 152822f9-20f7-4fe9-ad9f-bae31945e981
md"""Next, we draw from the posterior using [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl)."""

# ╔═╡ 2c44f500-c102-4a2e-980e-42afcfbba5bb
post = map(1:nchains) do _
    dynamicHMC(rng3, param_mod_soss, (y=y,), ndraws)
end

# ╔═╡ e2260ee1-c792-4b74-97c4-bbf6803a56d7
md"Finally, we update the posterior samples with draws from the posterior predictive distribution."

# ╔═╡ 0219a0d0-13ce-43d0-a048-8e95e55500db
begin
    pred = predictive(model_soss, :μ, :τ, :θ)
    post_postpred = map(post) do post_draws
        map(post_draws) do post_draw
            pred_draw = rand(rng3, pred(post_draw))
            pred_draw = delete(pred_draw, keys(constant_data))
            return merge(pred_draw, post_draw)
        end
    end
end

# ╔═╡ 500c4b4d-25eb-4f27-8e8b-7287bfeddf92
md"""
Each Soss draw is a `NamedTuple`.
We can plot the rank order statistics of the posterior to identify poor convergence:
"""

# ╔═╡ eac7b059-129d-472b-a69e-b1611c7cc703
begin
    plot_rank(post; var_names=["μ", "τ"])
    gcf()
end

# ╔═╡ 17d9fff5-d8f6-42c5-8cd1-70ecb34084c7
md"Now we combine all of the samples to an `InferenceData`:"

# ╔═╡ ddb1373a-f180-4efc-9193-905d03be4d8a
idata_soss = from_namedtuple(
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

# ╔═╡ bfa2c30f-f50c-45de-bd2a-da199ce67cfb
md"We can compare the prior and posterior predictive distributions:"

# ╔═╡ e56ee0c8-bc4a-4b37-bfcc-9ca1c440c1f3
begin
    plot_density(
        [idata_soss.posterior_predictive, idata_soss.prior_predictive];
        data_labels=["Post-pred", "Prior-pred"],
        var_names=["y"],
    )
    gcf()
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ArviZ = "131c737c-5715-5e2e-ad31-c244f01c1dc7"
CmdStan = "593b3428-ca2f-500c-ae53-031589ec8ddd"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
NamedTupleTools = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Soss = "8ce77f84-9b61-11e8-39ff-d17a774bf41c"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
ArviZ = "~0.5.4"
CmdStan = "~6.2.0"
Distributions = "~0.23.8"
MCMCChains = "~4.12.0"
NamedTupleTools = "~0.13.7"
PlutoUI = "~0.7.1"
PyPlot = "~2.9.0"
Soss = "~0.14.4"
Turing = "~0.16.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractLattices]]
git-tree-sha1 = "f35684b7349da49fcc8a9e520e30e45dbb077166"
uuid = "398f06c4-4d28-53ec-89ca-5b2656b7603d"
version = "0.2.1"

[[AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "21279159f6be4b2fd00e1a4a1f736893100408fc"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "3.2.0"

[[AbstractPPL]]
deps = ["AbstractMCMC"]
git-tree-sha1 = "ba9984ea1829e16b3a02ee49497c84c9795efa25"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.1.4"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[AdvancedHMC]]
deps = ["ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "Parameters", "ProgressMeter", "Random", "Requires", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7e85ed4917716873423f8d47da8c1275f739e0b7"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.2.27"

[[AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "Random", "Requires"]
git-tree-sha1 = "31ca445cb072fc347fe01548377789d6d64d2498"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.6.1"

[[AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Libtask", "Random", "StatsFuns"]
git-tree-sha1 = "22bf49efbdc7adb01c1f2e56489ae3e5752ee969"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.2.2"

[[AdvancedVI]]
deps = ["Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "130d6b17a3a9d420d9a6b37412cae03ffd6a64ff"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.1.3"

[[ArgCheck]]
git-tree-sha1 = "dedbbb2ddb876f899585c4ec4433265e3017215a"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.1.0"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "045ff5e1bc8c6fb1ecb28694abba0a0d55b5f4f5"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.17"

[[ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra"]
git-tree-sha1 = "951c3fc1ff93497c88fb1dfa893f4de55d0b38e3"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "0.3.8"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[ArviZ]]
deps = ["Conda", "DataFrames", "Markdown", "NamedTupleTools", "PyCall", "PyPlot", "REPL", "Requires", "StatsBase"]
git-tree-sha1 = "c2e545b2c1f4113ecc05a3b333309873c44bf52c"
uuid = "131c737c-5715-5e2e-ad31-c244f01c1dc7"
version = "0.5.4"

[[AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "f31f50712cbdf40ee8287f0443b57503e34122ef"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.3"

[[BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "d53b1eaefd48e233545d21f5b764c8ee54df4a09"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.30"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Statistics", "UUIDs"]
git-tree-sha1 = "068fda9b756e41e6c75da7b771e6f89fa8a43d15"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "0.7.0"

[[Bijectors]]
deps = ["ArgCheck", "Compat", "Distributions", "LinearAlgebra", "MappedArrays", "NNlib", "NonlinearSolve", "Random", "Reexport", "Requires", "SparseArrays", "Statistics", "StatsFuns"]
git-tree-sha1 = "66d899a108487588b62e04b939610667f0d19d80"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.8.12"

[[CSV]]
deps = ["Dates", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode"]
git-tree-sha1 = "b83aa3f513be680454437a0eee21001607e5d983"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.8.5"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[CanonicalTraits]]
deps = ["MLStyle"]
git-tree-sha1 = "f959d0e7164fb0262b02abecb93cf42b9a9f3188"
uuid = "a603d957-0e48-4f86-8fbd-0b7bc66df689"
version = "0.2.4"

[[CategoricalArrays]]
deps = ["DataAPI", "Future", "JSON", "Missings", "Printf", "Statistics", "StructTypes", "Unicode"]
git-tree-sha1 = "2ac27f59196a68070e132b25713f9a5bbc5fa0d2"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.8.3"

[[ChainRules]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "720fa9a9ce61ff18842a40f501d6a1f8ba771c64"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "0.8.6"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "d659e42240c2162300b321f05173cab5cc40a5ba"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "0.10.4"

[[CmdStan]]
deps = ["CSV", "DataFrames", "DelimitedFiles", "DocStringExtensions", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "cb153ba313f71db7d39bb516e80376a9d71f1842"
uuid = "593b3428-ca2f-500c-ae53-031589ec8ddd"
version = "6.2.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "e4e2b39db08f967cc1360951f01e8a75ec441cab"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.30.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[CompositionsBase]]
git-tree-sha1 = "f3955eb38944e5dd0fabf8ca1e267d94941d34a5"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.0"

[[Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

[[ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1dc43957fb9a1574fa1b7a449e101bd1fd3a9fb7"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.2.1"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "dfb3b7e89e395be1e25c2ad6d7690dc29cc53b1d"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.6.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "66ee4fe515a9294a8836ef18eea7239c6ac3db5e"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.1.1"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DefineSingletons]]
git-tree-sha1 = "77b4ca280084423b728662fe040e5ff8819347c5"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.1"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffEqDiffTools]]
deps = ["LinearAlgebra", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "b992345a39b4d9681342ae795a8dacc100730182"
uuid = "01453d9d-ee7c-5054-8395-0335cb756afa"
version = "0.14.0"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "214c3fcac57755cfda163d91c58893a8723f93e9"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.0.2"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "9c41285c57c6e0d73a21ed4b65f6eec34805f937"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.23.8"

[[DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "DiffRules", "Distributions", "FillArrays", "LinearAlgebra", "NaNMath", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "1c0ef4fe9eaa9596aca50b15a420e987b8447e56"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.28"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[DynamicHMC]]
deps = ["ArgCheck", "DocStringExtensions", "LinearAlgebra", "LogDensityProblems", "NLSolversBase", "Optim", "Parameters", "ProgressMeter", "Random", "Statistics"]
git-tree-sha1 = "7aa21d9ff8d2dafb8a4bf9f1b18c69bcc8960f8d"
uuid = "bbc10e6e-7c05-544b-b16e-64fede858acb"
version = "2.2.0"

[[DynamicPPL]]
deps = ["AbstractMCMC", "AbstractPPL", "Bijectors", "ChainRulesCore", "Distributions", "MacroTools", "Random", "ZygoteRules"]
git-tree-sha1 = "5121b72cbe2f92558754ad601a6af33c2bc5fdbe"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.12.1"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "8041575f021cba5a099a456b4163c9a08b566a02"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.0"

[[EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterface", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "764cddab41cd15f127767855722f1bf54b49c64a"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "0.4.3"

[[ExprTools]]
git-tree-sha1 = "10407a39b87f29d47ebaca8edbc75d7c302ff93e"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.3"

[[EzXML]]
deps = ["Printf", "XML2_jll"]
git-tree-sha1 = "0fa3b52a04a4e210aeb1626def9c90df3ae65268"
uuid = "8f5d6c58-4d21-5cfd-889c-e3ad7ee6a615"
version = "1.1.0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "4863cbb7910079369e258dee4add9d06ead5063a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.8.14"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "f6f80c8f934efd49a286bb5315360be66956dfc4"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.8.0"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[FlexLinearAlgebra]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "0a9c6dac1dcd94b496add97faf72bad9cd5bd4de"
uuid = "b67e1e5a-d13e-5892-ad81-fb75f5903773"
version = "0.1.0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "e2af66012e08966366a43251e1fd421522908be6"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.18"

[[FunctionWrappers]]
git-tree-sha1 = "241552bc2209f0fa068b6415b1942cc0aa486bcc"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.2"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GeneralizedGenerated]]
deps = ["CanonicalTraits", "DataStructures", "JuliaVariables", "MLStyle"]
git-tree-sha1 = "7dd404baf79b28f117917633f0cc1d2976c1fd9f"
uuid = "6b9d7cbe-bcb9-11e9-073f-15a7a543e2eb"
version = "0.2.8"

[[Graphs]]
deps = ["DataStructures", "SparseArrays"]
git-tree-sha1 = "9409e40f53532c45f2478c33531aa7a65ec4e2de"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "0.10.3"

[[Hwloc]]
deps = ["Hwloc_jll"]
git-tree-sha1 = "92d99146066c5c6888d5a3abc871e6a214388b91"
uuid = "0e44f5e4-bd66-52a0-8798-143a42290a1d"
version = "2.0.0"

[[Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aac91e34ef4c166e0857e3d6052a3467e5732ceb"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.4.1+0"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[InitialValues]]
git-tree-sha1 = "26c8832afd63ac558b98a823265856670d898b6c"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.2.10"

[[InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[Intervals]]
deps = ["Dates", "Printf", "RecipesBase", "Serialization", "TimeZones"]
git-tree-sha1 = "323a38ed1952d30586d0fe03412cde9399d3618b"
uuid = "d8418881-c3e1-53bb-8760-2df7ec849ed5"
version = "1.5.0"

[[InvertedIndices]]
deps = ["Test"]
git-tree-sha1 = "15732c475062348b0165684ffe28e85ea8396afc"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.0.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1a8c6237e78b714e901e406c096fc8a65528af7d"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.1"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "81690084b6198a2e1da36fcfda16eeca9f9f24e4"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.1"

[[JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "MatrixFactorizations", "StaticArrays"]
git-tree-sha1 = "e0b604d3b6da2a6e9e91c6cf928f79d2092619f3"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "0.16.16"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "71be1eb5ad19cb4f61fa8c73395c0338fd092ae0"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.2"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8d22e127ea9a0917bc98ebd3755c8bd31989381e"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+0"

[[Libtask]]
deps = ["Libtask_jll", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "6088b80fb5017440579ea8113a516ad2807afe19"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.5.1"

[[Libtask_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "901fc8752bbc527a6006a951716d661baa9d54e9"
uuid = "3ae2931a-708c-5973-9c38-ccf7496fb450"
version = "0.4.3+0"

[[LightXML]]
deps = ["Libdl", "XML2_jll"]
git-tree-sha1 = "e129d9391168c677cd4800f5c0abb1ed8cb3794f"
uuid = "9c8b4983-aa76-5018-a973-4c85ecc9e179"
version = "0.9.0"

[[LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogDensityProblems]]
deps = ["ArgCheck", "BenchmarkTools", "DiffResults", "DocStringExtensions", "Random", "Requires", "TransformVariables", "UnPack"]
git-tree-sha1 = "e3600cd2468d2b5356c240a5d27e0ef48fd451a3"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "0.10.5"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "1ba664552f1ef15325e68dc4c05c3ef8c2d5d885"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.2.4"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "59b45fd91b743dff047313bb7af0f84167aef80d"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.6"

[[LoopVectorization]]
deps = ["ArrayInterface", "DocStringExtensions", "IfElse", "LinearAlgebra", "OffsetArrays", "Polyester", "Requires", "SLEEFPirates", "Static", "StrideArraysCore", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "b16dde45ba9e2506358d4d7fe13f746330e8e622"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.37"

[[MCMCChains]]
deps = ["AbstractFFTs", "AbstractMCMC", "AxisArrays", "Compat", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "LinearAlgebra", "MLJModelInterface", "NaturalSort", "PrettyTables", "Random", "RecipesBase", "Serialization", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "68b58fa78123cb38b2fd1394e8aff6d35b22de4f"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "4.12.0"

[[MLJModelInterface]]
deps = ["Random", "ScientificTypes", "StatisticalTraits"]
git-tree-sha1 = "cafa0e923ce1ae659a4b4cb8eb03c98b916f0d4d"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.1.0"

[[MLStyle]]
git-tree-sha1 = "594e189325f66e23a8818e5beb11c43bb0141bcd"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.10"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "6a8a2a625ab0dea913aba95c11370589e0239ff0"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.6"

[[MappedArrays]]
deps = ["FixedPointNumbers"]
git-tree-sha1 = "b92bd220c95a8bbe89af28f11201fd080e0e3fe7"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.3.0"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MatrixFactorizations]]
deps = ["ArrayLayouts", "LinearAlgebra", "Random"]
git-tree-sha1 = "f609fbed57eede4fd9c8529f59b3d30454f9e5ca"
uuid = "a3b82374-2e81-5b9e-98ce-41277c0e4c87"
version = "0.5.2"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[MicroCollections]]
deps = ["BangBang", "Setfield"]
git-tree-sha1 = "e991b6a9d38091c4a0d7cd051fcb57c05f98ac03"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.0"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f8c673ccc215eb50fcadb285f522420e29e69e1c"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "0.4.5"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Mocking]]
deps = ["ExprTools"]
git-tree-sha1 = "916b850daad0d46b8c71f65f719c49957e9513ed"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.1"

[[MonteCarloMeasurements]]
deps = ["Distributed", "Distributions", "LinearAlgebra", "MacroTools", "Random", "RecipesBase", "Requires", "SLEEFPirates", "StaticArrays", "Statistics", "StatsBase", "Test"]
git-tree-sha1 = "6878d5c392d83117148b83e65b5430260d078cb9"
uuid = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
version = "0.9.15"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NLSolversBase]]
deps = ["Calculus", "DiffEqDiffTools", "DiffResults", "Distributed", "ForwardDiff"]
git-tree-sha1 = "f1b8ed89fa332f410cfc7c937682eb4d0b361521"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.5.0"

[[NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "0bf1fbb9dc557f2af9fb7e1337366d69de0dc78c"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.7.21"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "9ba8ddb0c06a08b1bad81b7120d13288e5d766fa"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.9.5"

[[NamedTupleTools]]
git-tree-sha1 = "63831dcea5e11db1c0925efe5ef5fc01d528c522"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.13.7"

[[NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[NonlinearSolve]]
deps = ["ArrayInterface", "FiniteDiff", "ForwardDiff", "IterativeSolvers", "LinearAlgebra", "RecursiveArrayTools", "RecursiveFactorization", "Reexport", "SciMLBase", "Setfield", "StaticArrays", "UnPack"]
git-tree-sha1 = "ef18e47df4f3917af35be5e5d7f5d97e8a83b0ec"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "0.3.8"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "1381a7142eefd4cd12f052a4d2d790fe21bd1d55"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.9.2"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Optim]]
deps = ["Compat", "FillArrays", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "c05aa6b694d426df87ff493306c1c5b4b215e148"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "0.22.0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse", "Test"]
git-tree-sha1 = "95a4038d1011dfdbde7cecd2ad0ac411e53ab1bc"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.10.1"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "2276ac65f1e236e0a6ea70baff3f62ad4c625345"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.2"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "c8abc88faa3f7a3950832ac5d6e690881590d6dc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.0"

[[Permutations]]
deps = ["Combinatorics", "LinearAlgebra", "Random"]
git-tree-sha1 = "0fb6a7cd0e0e6f68e541efee2aaec899a8dbdcd1"
uuid = "2ae35dd2-176d-5d53-8349-f30d82d94d4f"
version = "0.4.4"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "Logging", "Markdown", "Random", "Suppressor"]
git-tree-sha1 = "45ce174d36d3931cd4e37a47f93e07d1455f038d"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.1"

[[Polyester]]
deps = ["ArrayInterface", "IfElse", "Requires", "Static", "StrideArraysCore", "ThreadingUtilities", "VectorizationBase"]
git-tree-sha1 = "04a03d3f8ae906f4196b9085ed51506c4b466340"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.3.1"

[[Polynomials]]
deps = ["Intervals", "LinearAlgebra", "OffsetArrays", "RecipesBase"]
git-tree-sha1 = "0b15f3597b01eb76764dd03c3c23d6679a3c32c8"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "1.2.1"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "cde4ce9d6f33219465b55162811d8de8139c0414"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.2.1"

[[PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Primes]]
git-tree-sha1 = "afccf037da52fa596223e5a0e331ff752e0e845c"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "169bb8ea6b1b143c5cf57df6d34d022a7b60c6db"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.92.3"

[[PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "67dde2482fe1a72ef62ed93f8c239f947638e5a2"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.9.0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[RecipesBase]]
git-tree-sha1 = "b3fb709f3c97bfc6e948be68beeecb55a0b340ae"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.1"

[[RecursiveArrayTools]]
deps = ["ArrayInterface", "DocStringExtensions", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "b3f4e34548b3d3d00e5571fd7bc0a33980f01571"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.11.4"

[[RecursiveFactorization]]
deps = ["LinearAlgebra", "LoopVectorization"]
git-tree-sha1 = "9514a935538cd568befe8520752c2fb0eef857af"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.1.12"

[[Reexport]]
deps = ["Pkg"]
git-tree-sha1 = "7b1d07f411bc8ddb7977ec7f377b97b158514fe0"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "0.2.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[ReverseDiff]]
deps = ["DiffResults", "DiffRules", "ForwardDiff", "FunctionWrappers", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "SpecialFunctions", "StaticArrays", "Statistics"]
git-tree-sha1 = "63ee24ea0689157a1113dbdab10c6cb011d519c4"
uuid = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
version = "1.9.0"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "18e8cd8007e9b85e2602dafc6eea22d4c1a0aec4"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.21"

[[SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "05aa1ee0b6f0c875b0d6572a77c57225e47b688f"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.13.4"

[[ScientificTypes]]
git-tree-sha1 = "b4e89a674804025c4a5843e35e562910485690c2"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "1.1.2"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "bc967c221ccdb0b85511709bda96ee489396f544"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.2"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "d5640fc570fb1b6c54512f0bd3853866bd298b3e"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.7.0"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[SimpleGraphs]]
deps = ["AbstractLattices", "Combinatorics", "DataStructures", "IterTools", "LightXML", "LinearAlgebra", "Optim", "Polynomials", "Primes", "Random", "SimplePartitions", "SimpleRandom", "SimpleTools", "SparseArrays", "Statistics"]
git-tree-sha1 = "d9b508600279433c2840b8f96c684ddbed281925"
uuid = "55797a34-41de-5266-9ec1-32ac4eb504d3"
version = "0.4.4"

[[SimplePartitions]]
deps = ["AbstractLattices", "DataStructures", "Permutations"]
git-tree-sha1 = "dcc02923a53f316ab97da8ef3136e80b4543dbf1"
uuid = "ec83eff0-a5b5-5643-ae32-5cbf6eedec9d"
version = "0.3.0"

[[SimplePosets]]
deps = ["FlexLinearAlgebra", "Primes", "SimpleGraphs", "SimplePartitions"]
git-tree-sha1 = "929620ab51cce723a222240dfeb7b9dbe7631f71"
uuid = "b2aef97b-4721-5af9-b440-0bad754dc5ba"
version = "0.1.3"

[[SimpleRandom]]
deps = ["Distributions", "LinearAlgebra", "Random"]
git-tree-sha1 = "59a830985c7292e4028955b631d3f5edcf8ff35e"
uuid = "a6525b86-64cd-54fa-8f65-62fc48bdc0e8"
version = "0.2.1"

[[SimpleTools]]
deps = ["Polynomials"]
git-tree-sha1 = "ed7c851fe8355f8ac68b514826c53367bb05e23c"
uuid = "4696fa5f-36f0-5b18-99a6-fef83351280f"
version = "0.4.1"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "2ec1962eba973f383239da22e75218565c390a96"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.0"

[[Soss]]
deps = ["AdvancedHMC", "Bijectors", "CategoricalArrays", "DiffResults", "Distributions", "DistributionsAD", "DynamicHMC", "FillArrays", "ForwardDiff", "GeneralizedGenerated", "Graphs", "IterTools", "LazyArrays", "LogDensityProblems", "MLStyle", "MacroTools", "MonteCarloMeasurements", "NamedTupleTools", "Printf", "Random", "RecipesBase", "Reexport", "Requires", "ReverseDiff", "Setfield", "SimpleGraphs", "SimplePartitions", "SimplePosets", "SpecialFunctions", "Statistics", "StatsFuns", "TransformVariables"]
git-tree-sha1 = "3a028d831a9de4f4b28c176b8830119165bf2eaf"
uuid = "8ce77f84-9b61-11e8-39ff-d17a774bf41c"
version = "0.14.4"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["OpenSpecFun_jll"]
git-tree-sha1 = "d8d8b8a9f4119829410ecd706da4cc8594a1e020"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "0.10.3"

[[SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "edef25a158db82f4940720ebada14a60ef6c4232"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.13"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "2740ea27b66a41f9d213561a04573da5d3823d4b"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.2.5"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "da4cf579416c81994afd6322365d00916c79b8ae"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "0.12.5"

[[StatisticalTraits]]
deps = ["ScientificTypes"]
git-tree-sha1 = "2d882a163c295d5d754e4102d92f4dda5a1f906b"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "1.1.0"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2f6792d523d7448bbe2fec99eca9218f06cc746d"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.8"

[[StatsFuns]]
deps = ["LogExpFunctions", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "30cd8c360c54081f806b1ee14d2eecbef3c04c49"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.8"

[[StrideArraysCore]]
deps = ["ArrayInterface", "Requires", "ThreadingUtilities", "VectorizationBase"]
git-tree-sha1 = "efcdfcbb8cf91e859f61011de1621be34b550e69"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.1.13"

[[StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "e36adc471280e8b346ea24c5c87ba0571204be7a"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.7.2"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "aa30f8bb63f9ff3f8303a06c604c8500a69aa791"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.4.3"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "e185a19bb9172f0cf5bc71233fab92a46f7ae154"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.3"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[ThreadingUtilities]]
deps = ["VectorizationBase"]
git-tree-sha1 = "28f4295cd761ce98db2b5f8c1fe6e5c89561efbe"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.4.4"

[[TimeZones]]
deps = ["Dates", "EzXML", "LazyArtifacts", "Mocking", "Pkg", "Printf", "RecipesBase", "Serialization", "Unicode"]
git-tree-sha1 = "960099aed321e05ac649c90d583d59c9309faee1"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.5.5"

[[Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "LinearAlgebra", "MacroTools", "NNlib", "NaNMath", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "bf4adf36062afc921f251af4db58f06235504eff"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.16"

[[Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "34f27ac221cb53317ab6df196f9ed145077231ff"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.65"

[[TransformVariables]]
deps = ["ArgCheck", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "Parameters", "Pkg", "Random"]
git-tree-sha1 = "05cbdc6c521a03d5c258b682eb4f7e04d20991ba"
uuid = "84d833dd-6860-57f9-a1a7-6da5db126cff"
version = "0.3.12"

[[TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[Turing]]
deps = ["AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "DataStructures", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tracker", "ZygoteRules"]
git-tree-sha1 = "80ca4cd497ab7eb2ea1ea005af3d45509c8debe3"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.16.2"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VectorizationBase]]
deps = ["ArrayInterface", "Hwloc", "IfElse", "Libdl", "LinearAlgebra", "Static"]
git-tree-sha1 = "7c8974c7de377a2dc67e778017c78f96fc8f0fc6"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.20.16"

[[VersionParsing]]
git-tree-sha1 = "80229be1f670524750d905f8fc8148e5a8c4537f"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.0"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "9e7a1e8ca60b742e508a315c17eef5211e7fbfd7"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─4d0e37f3-85f5-4ad8-bbba-8e5a3288f48b
# ╟─d2eedd48-48c6-4fcd-b179-6be7fe68d3d6
# ╠═467c2d13-6bfe-4feb-9626-fb14796168aa
# ╠═06b00794-e97f-472b-b526-efe4815103f8
# ╟─5acbfd9a-cfea-4c3c-a3f0-1744eb7e4e27
# ╠═efb3f0af-9fac-48d8-bbb2-2dd6ebd5e4f6
# ╠═401e9b91-0bca-4369-8d36-3d9f0b3ad60b
# ╟─2c718ea5-2800-4df6-b62c-e0a9e440a1c3
# ╠═49f19c17-ac1d-46b5-a655-4376b7713244
# ╟─a9789109-2b90-40f7-926c-e7c87025d15f
# ╠═69124a94-a6aa-43f6-8d4f-fa9a6b1cfef1
# ╟─10986e0e-55f7-40ea-ab63-274fbab3126d
# ╠═f383d541-e22d-44b4-b8cb-28b3d67944a1
# ╠═85bbcba7-c0f9-4c86-9cdf-a27055d3d448
# ╟─bd4ab044-51ce-4af9-83b2-bd8fc827f810
# ╠═500f4e0d-0a36-4b5c-8900-667560fbf1d4
# ╟─1129ad94-f65a-4332-b354-21bcf7e53541
# ╠═803efdd8-656e-4e37-ba36-81195d064972
# ╟─79f342c8-0738-432b-bfd7-2da25e50fa91
# ╠═6209f947-5001-4507-b3e8-9747256f3328
# ╟─26692722-db0d-41de-b58c-339b639f6948
# ╠═14046b83-9c0a-4d33-ae4e-36c7d6f1b2e6
# ╟─737f319c-1ddd-45f2-8d10-aaecdc1334be
# ╠═5be3131d-db8b-44c8-9fcc-571d68695148
# ╟─0787a842-63ca-407d-9d07-a3d949940f92
# ╠═6e8343c8-bee3-4e1d-82d6-1885bfd1dbec
# ╟─cba6f6c9-82c4-4957-acc3-36e9f1c95d76
# ╠═79374a8b-98c1-443c-acdd-0ee00bd42b38
# ╟─c23562ba-3b0b-49d9-bb41-2fedaa9e9500
# ╠═d0b447fe-26b2-48e0-bcec-d9f04697973a
# ╟─4d2fdcbe-c1d4-43b6-b382-f2e956b952a1
# ╠═5a075722-232f-40fc-a499-8dc5b0c2424a
# ╟─1b5af2c3-f2ce-4e9d-9ad7-ac287a9178e2
# ╠═b38c7a43-f00c-43c0-aa6b-9c581d6d0c73
# ╟─a3b71e4e-ac1f-4404-b8a2-35d03d326774
# ╠═f552b5b5-9744-41df-af90-46405367ea0b
# ╟─9d3673f5-b57b-432e-944a-70b23643128a
# ╠═05c9be29-7758-4324-971c-5579f99aaf9d
# ╟─98acc304-22e3-4e6b-a2f4-d22f6847145b
# ╠═b46af168-1ce3-4058-a014-b66c645a6e0d
# ╠═ab145e41-b230-4cad-bef5-f31e0e0770d4
# ╟─ffc7730c-d861-48e8-b173-b03e0542f32b
# ╠═020cbdc0-a0a2-4d20-838f-c99b541d5832
# ╟─e44b260c-9d2f-43f8-a64b-04245a0a5658
# ╠═5070bbbc-68d2-49b8-bd91-456dc0da4573
# ╟─2674d532-a337-471e-8ba7-02b430f49f12
# ╠═14408abe-a16f-4cc0-a6f3-0bb2645653b7
# ╠═9daec35c-3d6e-443c-87f9-213d51964f75
# ╠═6a78e4a8-86c4-4438-b9bb-7c433d2bc8c8
# ╟─a94b3ad6-2305-43f6-881f-d871286b906a
# ╠═d385ceea-beb5-4ec7-b50d-be691266440b
# ╟─152822f9-20f7-4fe9-ad9f-bae31945e981
# ╠═2c44f500-c102-4a2e-980e-42afcfbba5bb
# ╟─e2260ee1-c792-4b74-97c4-bbf6803a56d7
# ╠═0219a0d0-13ce-43d0-a048-8e95e55500db
# ╟─500c4b4d-25eb-4f27-8e8b-7287bfeddf92
# ╠═eac7b059-129d-472b-a69e-b1611c7cc703
# ╟─17d9fff5-d8f6-42c5-8cd1-70ecb34084c7
# ╠═ddb1373a-f180-4efc-9193-905d03be4d8a
# ╟─bfa2c30f-f50c-45de-bd2a-da199ce67cfb
# ╠═e56ee0c8-bc4a-4b37-bfcc-9ca1c440c1f3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
