### A Pluto.jl notebook ###
# v0.19.16

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
using ArviZ, Distributions, LinearAlgebra, PyPlot, Random, StanSample, Turing

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

ArviZ.jl is designed to be used with libraries like [Stan](https://github.com/StanJulia/Stan.jl), [Turing.jl](https://turing.ml), and [Soss.jl](https://github.com/cscherrer/Soss.jl) but works fine with raw arrays.
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
    s = (50, 10)
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
    (; y=cat(log_likelihood_y...; dims=3))
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
## [Plotting with Stan.jl outputs](#Plotting-with-Stan.jl-outputs)

StanSample.jl comes with built-in support for producing `InferenceData` outputs.

Here is the same centered eight schools model in Stan:
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
    idata_stan = mktempdir() do path
        stan_model = SampleModel("schools", schools_code, path)
        _ = stan_sample(
            stan_model;
            data=schools_data,
            num_chains=nchains,
            num_warmups=ndraws_warmup,
            num_samples=ndraws,
            seed=28983,
            summary=false,
        )
        return StanSample.inferencedata(
            stan_model;
            posterior_predictive_var=:y_hat,
            observed_data=(; y),
            log_likelihood_var=:log_lik,
            coords=(; school=schools),
            dims=NamedTuple(
                k => (:school,) for k in (:y, :sigma, :theta, :log_lik, :y_hat)
            ),
        )
    end
end;

# ╔═╡ ab145e41-b230-4cad-bef5-f31e0e0770d4
begin
    plot_density(idata_stan; var_names=(:mu, :tau))
    gcf()
end

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
# ╟─98acc304-22e3-4e6b-a2f4-d22f6847145b
# ╠═b46af168-1ce3-4058-a014-b66c645a6e0d
# ╠═ab145e41-b230-4cad-bef5-f31e0e0770d4
# ╟─e44b260c-9d2f-43f8-a64b-04245a0a5658
# ╠═5070bbbc-68d2-49b8-bd91-456dc0da4573
# ╟─ac2b4378-bd1c-4164-af05-d9a35b1bb08f
# ╠═56a39a90-0594-48f4-ba04-f7b612019cd1
# ╠═2c0f3d74-8b1e-4648-b0fa-f8050035b0fd
# ╠═fd84237a-7a19-4bbb-a702-faa31075ecbc
# ╠═ad29b7f3-6f5e-4b04-bf70-2308a7d110c0
