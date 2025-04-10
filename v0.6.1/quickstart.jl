### A Pluto.jl notebook ###
# v0.19.5

using Markdown
using InteractiveUtils

# ╔═╡ 2c0f3d74-8b1e-4648-b0fa-f8050035b0fd
using Pkg, InteractiveUtils

# ╔═╡ ac57d957-0bdd-457a-ac15-9a4f94f0c785
# Remove this cell to use release versions of dependencies 
# hideall
let
    docs_dir = dirname(@__DIR__)
    pkg_dir = dirname(docs_dir)

    Pkg.activate(docs_dir)
    Pkg.develop(; path=pkg_dir)
    Pkg.instantiate()
end;

# ╔═╡ 467c2d13-6bfe-4feb-9626-fb14796168aa
begin
    using ArviZ, CmdStan, Distributions, LinearAlgebra, PyPlot, Random, Soss, Turing
    using Soss.MeasureTheory: HalfCauchy
    using SampleChainsDynamicHMC: getchains, dynamichmc
end

# ╔═╡ 56a39a90-0594-48f4-ba04-f7b612019cd1
using PlutoUI

# ╔═╡ a23dfd65-50a8-4872-8c41-661e96585aca
md"""
# [ArviZ.jl Quickstart](#quickstart)

!!! note
    
    This tutorial is adapted from [ArviZ's quickstart](https://python.arviz.org/en/latest/getting_started/Introduction.html).
"""

# ╔═╡ d2eedd48-48c6-4fcd-b179-6be7fe68d3d6
md"""
## [Setup](#setup)

Here we add the necessary packages for this notebook and load a few we will use throughout.
"""

# ╔═╡ 06b00794-e97f-472b-b526-efe4815103f8
# ArviZ ships with style sheets!
ArviZ.use_style("arviz-darkgrid")

# ╔═╡ 5acbfd9a-cfea-4c3c-a3f0-1744eb7e4e27
md"""
## [Get started with plotting](#Get-started-with-plotting)

ArviZ.jl is designed to be used with libraries like [CmdStan](https://github.com/StanJulia/CmdStan.jl), [Turing.jl](https://turing.ml), and [Soss.jl](https://github.com/cscherrer/Soss.jl) but works fine with raw arrays.
"""

# ╔═╡ efb3f0af-9fac-48d8-bbb2-2dd6ebd5e4f6
rng1 = Random.MersenneTwister(37772);

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
let
    s = (10, 50)
    plot_forest((
        normal=randn(rng1, s),
        gumbel=rand(rng1, Gumbel(), s),
        student_t=rand(rng1, TDist(6), s),
        exponential=rand(rng1, Exponential(), s),
    ),)
    gcf()
end

# ╔═╡ a9789109-2b90-40f7-926c-e7c87025d15f
md"""
## [Plotting with MCMCChains.jl's `Chains` objects produced by Turing.jl](#Plotting-with-MCMCChains.jl's-Chains-objects-produced-by-Turing.jl)

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

# ╔═╡ 86cb5e19-49e4-4e5e-8b89-e76936932055
rng2 = Random.MersenneTwister(16653);

# ╔═╡ 85bbcba7-c0f9-4c86-9cdf-a27055d3d448
begin
    param_mod_turing = model_turing(y, σ)
    sampler = NUTS(ndraws_warmup, 0.8)

    turing_chns = Turing.sample(
        rng2, model_turing(y, σ), sampler, MCMCThreads(), ndraws, nchains
    )
end;

# ╔═╡ bd4ab044-51ce-4af9-83b2-bd8fc827f810
md"""
Most ArviZ functions work fine with `Chains` objects from Turing:
"""

# ╔═╡ 500f4e0d-0a36-4b5c-8900-667560fbf1d4
begin
    plot_autocorr(turing_chns; var_names=(:μ, :τ))
    gcf()
end

# ╔═╡ 1129ad94-f65a-4332-b354-21bcf7e53541
md"""
### Convert to `InferenceData`

For much more powerful querying, analysis and plotting, we can use built-in ArviZ utilities to convert `Chains` objects to multidimensional data structures with named dimensions and indices.
Note that for such dimensions, the information is not contained in `Chains`, so we need to provide it.

ArviZ is built to work with [`InferenceData`](https://julia.arviz.org/stable/reference/#ArviZ.InferenceData), and the more *groups* it has access to, the more powerful analyses it can perform.
"""

# ╔═╡ 803efdd8-656e-4e37-ba36-81195d064972
idata_turing_post = from_mcmcchains(
    turing_chns;
    coords=(; school=schools),
    dims=NamedTuple(k => (:school,) for k in (:y, :σ, :θ)),
    library="Turing",
)

# ╔═╡ 79f342c8-0738-432b-bfd7-2da25e50fa91
md"""
Each group is an [`ArviZ.Dataset`](https://julia.arviz.org/stable/reference/#ArviZ.Dataset), a `DimensionalData.AbstractDimStack` that can be used identically to a [`DimensionalData.Dimstack`](https://rafaqz.github.io/DimensionalData.jl/stable/api/#DimensionalData.DimStack).
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

With a few more steps, we can use Turing to compute additional useful groups to add to the `InferenceData`.

To sample from the prior, one simply calls `sample` but with the `Prior` sampler:
"""

# ╔═╡ 79374a8b-98c1-443c-acdd-0ee00bd42b38
prior = Turing.sample(rng2, param_mod_turing, Prior(), ndraws);

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
And to extract the pointwise log-likelihoods, which is useful if you want to compute metrics such as [`loo`](https://julia.arviz.org/stable/reference/#ArviZ.loo),
"""

# ╔═╡ 5a075722-232f-40fc-a499-8dc5b0c2424a
log_likelihood = let
    log_likelihood = Turing.pointwise_loglikelihoods(
        param_mod_turing, MCMCChains.get_sections(turing_chns, :parameters)
    )
    # Ensure the ordering of the loglikelihoods matches the ordering of `posterior_predictive`
    ynames = string.(keys(posterior_predictive))
    log_likelihood_y = getindex.(Ref(log_likelihood), ynames)
    # Reshape into `(nchains, ndraws, size(y)...)`
    (; y=permutedims(cat(log_likelihood_y...; dims=3), (2, 1, 3)))
end;

# ╔═╡ 1b5af2c3-f2ce-4e9d-9ad7-ac287a9178e2
md"This can then be included in the [`from_mcmcchains`](https://julia.arviz.org/stable/reference/#ArviZ.from_mcmcchains) call from above:"

# ╔═╡ b38c7a43-f00c-43c0-aa6b-9c581d6d0c73
idata_turing = from_mcmcchains(
    turing_chns;
    posterior_predictive,
    log_likelihood,
    prior,
    prior_predictive,
    observed_data=(; y),
    coords=(; school=schools),
    dims=NamedTuple(k => (:school,) for k in (:y, :σ, :θ)),
    library=Turing,
)

# ╔═╡ a3b71e4e-ac1f-4404-b8a2-35d03d326774
md"""
Then we can for example compute the expected *leave-one-out (LOO)* predictive density, which is an estimate of the out-of-distribution predictive fit of the model:
"""

# ╔═╡ f552b5b5-9744-41df-af90-46405367ea0b
loo(idata_turing; pointwise=false) # higher is better

# ╔═╡ 9d3673f5-b57b-432e-944a-70b23643128a
md"""
If the model is well-calibrated, i.e. it replicates the true generative process well, the CDF of the pointwise LOO values should be similarly distributed to a uniform distribution.
This can be inspected visually:
"""

# ╔═╡ 05c9be29-7758-4324-971c-5579f99aaf9d
begin
    plot_loo_pit(idata_turing; y=:y, ecdf=true)
    gcf()
end

# ╔═╡ 98acc304-22e3-4e6b-a2f4-d22f6847145b
md"""
## [Plotting with CmdStan.jl outputs](#Plotting-with-CmdStan.jl-outputs)

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
            nchains,
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
    plot_density(stan_chns; var_names=(:mu, :tau))
    gcf()
end

# ╔═╡ ffc7730c-d861-48e8-b173-b03e0542f32b
md"""
Again, converting to `InferenceData`, we can get much richer labelling and mixing of data.
Note that we're using the same [`from_cmdstan`](https://julia.arviz.org/stable/reference/#ArviZ.from_cmdstan) function used by ArviZ to process cmdstan output files, but through the power of dispatch in Julia, if we pass a `Chains` object, it instead uses ArviZ.jl's overloads, which forward to `from_mcmcchains`.
"""

# ╔═╡ 020cbdc0-a0a2-4d20-838f-c99b541d5832
idata_stan = from_cmdstan(
    stan_chns;
    posterior_predictive=:y_hat,
    observed_data=(; y),
    log_likelihood=:log_lik,
    coords=(; school=schools),
    dims=NamedTuple(k => (:school,) for k in (:y, :sigma, :theta, :log_lik, :y_hat)),
)

# ╔═╡ e44b260c-9d2f-43f8-a64b-04245a0a5658
md"""Here is a plot showing where the Hamiltonian sampler had divergences:"""

# ╔═╡ 5070bbbc-68d2-49b8-bd91-456dc0da4573
begin
    plot_pair(
        idata_stan;
        coords=Dict(:school => ["Choate", "Deerfield", "Phillips Andover"]),
        divergences=true,
    )
    gcf()
end

# ╔═╡ 2674d532-a337-471e-8ba7-02b430f49f12
md"""
## [Plotting with Soss.jl outputs](#Plotting-with-Soss.jl-outputs)

With Soss, we can define our model for the posterior and easily use it to draw samples from the prior, prior predictive, posterior, and posterior predictive distributions.

First we define our model:
"""

# ╔═╡ 14408abe-a16f-4cc0-a6f3-0bb2645653b7
constant_data = (; J, σ);

# ╔═╡ 446341da-902e-474b-b6dc-b085ef74a99b
observed_data = (; y);

# ╔═╡ 9daec35c-3d6e-443c-87f9-213d51964f75
model_soss = Soss.@model (J, σ) begin
    μ ~ Soss.Normal(; μ=0, σ=5)
    τ ~ HalfCauchy(; σ=5)
    θ ~ Soss.Normal(; μ, σ=τ) |> iid(J)
    y ~ For(1:J) do j
        Soss.Normal(; μ=θ[j], σ=σ[j])
    end
end;

# ╔═╡ 6a78e4a8-86c4-4438-b9bb-7c433d2bc8c8
param_mod_soss = model_soss(; constant_data...)

# ╔═╡ a94b3ad6-2305-43f6-881f-d871286b906a
md"Then we draw from the prior and prior predictive distributions."

# ╔═╡ bfdfdff7-6551-4dd7-a7c8-69f8b57272ec
rng3 = MersenneTwister(5298);

# ╔═╡ d385ceea-beb5-4ec7-b50d-be691266440b
prior_priorpred = let Normal = Soss.Normal
    [rand(rng3, param_mod_soss, ndraws) for _ in 1:nchains]
end

# ╔═╡ 152822f9-20f7-4fe9-ad9f-bae31945e981
md"""Next, we draw from the posterior using [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl)."""

# ╔═╡ 2c44f500-c102-4a2e-980e-42afcfbba5bb
post = Soss.sample(rng3, param_mod_soss | observed_data, dynamichmc(), ndraws, nchains)

# ╔═╡ e2260ee1-c792-4b74-97c4-bbf6803a56d7
md"Finally, we update the posterior samples with draws from the posterior predictive distribution."

# ╔═╡ 0219a0d0-13ce-43d0-a048-8e95e55500db
postpred = let
    mod_pred = Soss.predictive(model_soss, :μ, :τ, :θ)
    map(getchains(post)) do chain
        [rand(rng3, mod_pred(; constant_data..., draw...)) for draw in chain]
    end
end

# ╔═╡ 500c4b4d-25eb-4f27-8e8b-7287bfeddf92
md"""
Each Soss draw is a `NamedTuple`.
We can plot the rank order statistics of the posterior to identify poor convergence:
"""

# ╔═╡ eac7b059-129d-472b-a69e-b1611c7cc703
begin
    plot_rank(post; var_names=(:μ, :τ))
    gcf()
end

# ╔═╡ 17d9fff5-d8f6-42c5-8cd1-70ecb34084c7
md"Now we combine all of the samples to an `InferenceData`:"

# ╔═╡ ddb1373a-f180-4efc-9193-905d03be4d8a
idata_soss = from_samplechains(
    post;
    posterior_predictive=postpred,
    prior=prior_priorpred,
    prior_predictive=(:y,),
    observed_data,
    constant_data=(; J=[J], σ),
    coords=(; school=schools),
    dims=NamedTuple(k => (:school,) for k in (:y, :σ, :θ)),
    library=Soss,
)

# ╔═╡ bfa2c30f-f50c-45de-bd2a-da199ce67cfb
md"We can compare the prior and posterior predictive distributions:"

# ╔═╡ e56ee0c8-bc4a-4b37-bfcc-9ca1c440c1f3
begin
    plot_density(
        [idata_soss.posterior_predictive, idata_soss.prior_predictive];
        data_labels=["Post-pred", "Prior-pred"],
        var_names=(:y,),
    )
    gcf()
end

# ╔═╡ ac2b4378-bd1c-4164-af05-d9a35b1bb08f
md"## [Environment](#environment)"

# ╔═╡ fd84237a-7a19-4bbb-a702-faa31075ecbc
with_terminal(Pkg.status; color=false)

# ╔═╡ ad29b7f3-6f5e-4b04-bf70-2308a7d110c0
with_terminal(versioninfo)

# ╔═╡ Cell order:
# ╟─a23dfd65-50a8-4872-8c41-661e96585aca
# ╟─d2eedd48-48c6-4fcd-b179-6be7fe68d3d6
# ╠═ac57d957-0bdd-457a-ac15-9a4f94f0c785
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
# ╠═86cb5e19-49e4-4e5e-8b89-e76936932055
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
# ╠═98acc304-22e3-4e6b-a2f4-d22f6847145b
# ╠═b46af168-1ce3-4058-a014-b66c645a6e0d
# ╠═ab145e41-b230-4cad-bef5-f31e0e0770d4
# ╟─ffc7730c-d861-48e8-b173-b03e0542f32b
# ╠═020cbdc0-a0a2-4d20-838f-c99b541d5832
# ╟─e44b260c-9d2f-43f8-a64b-04245a0a5658
# ╠═5070bbbc-68d2-49b8-bd91-456dc0da4573
# ╠═2674d532-a337-471e-8ba7-02b430f49f12
# ╠═14408abe-a16f-4cc0-a6f3-0bb2645653b7
# ╠═446341da-902e-474b-b6dc-b085ef74a99b
# ╠═9daec35c-3d6e-443c-87f9-213d51964f75
# ╠═6a78e4a8-86c4-4438-b9bb-7c433d2bc8c8
# ╟─a94b3ad6-2305-43f6-881f-d871286b906a
# ╠═bfdfdff7-6551-4dd7-a7c8-69f8b57272ec
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
# ╟─ac2b4378-bd1c-4164-af05-d9a35b1bb08f
# ╠═56a39a90-0594-48f4-ba04-f7b612019cd1
# ╠═2c0f3d74-8b1e-4648-b0fa-f8050035b0fd
# ╠═fd84237a-7a19-4bbb-a702-faa31075ecbc
# ╠═ad29b7f3-6f5e-4b04-bf70-2308a7d110c0
