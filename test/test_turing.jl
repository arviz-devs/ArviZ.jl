using Turing
using ArviZ
using ArviZ: groupnames
using Test
using Random

@testset "from_turing" begin
    nchains = 2
    ndraws = 10
    Turing.@model function demo(xs, y, n=length(xs))
        s ~ InverseGamma(2, 3)
        m ~ Normal(0, √s)
        for i in 1:n
            xs[i] ~ Normal(m, √s)
        end
        return y ~ Normal(m, √s)
    end
    xs = randn(5)
    y = randn()
    observed_data = (xs=xs, y=y)
    model = demo(observed_data...)
    chn = Turing.sample(
        model, Turing.MH(), Turing.MCMCThreads(), ndraws, nchains; progress=false
    )
    @test size(chn) == (ndraws, 3, nchains)

    idata1 = from_turing(chn)
    @test sort(groupnames(idata1)) == [:posterior, :sample_stats]
    @test idata1.posterior.inference_library == "Turing"
    VersionNumber(idata1.posterior.inference_library_version)

    idata2 = from_turing(; model=model, observed_data=false)
    @test sort(groupnames(idata2)) == [:prior, :sample_stats_prior]
    @test length(idata2.prior.chain.values) == 1
    @test length(idata2.prior.draw.values) == 1_000
    @test idata2.prior.inference_library == "Turing"
    VersionNumber(idata2.prior.inference_library_version)

    idata3 = from_turing(chn; model=model, observed_data=false)
    @test sort(groupnames(idata3)) ==
          sort([:posterior, :sample_stats, :prior, :sample_stats_prior])
    @test length(idata3.prior.chain.values) == nchains
    @test length(idata3.prior.draw.values) == ndraws

    idata4 = from_turing(chn; model=model, prior=false, observed_data=false)
    @test sort(groupnames(idata4)) == [:posterior, :sample_stats]

    idata5 = from_turing(chn; model=model, nchains=3, ndraws=100)
    @test idata5.posterior.inference_library == "Turing"
    VersionNumber(idata5.posterior.inference_library_version)
    @test sort(groupnames(idata5)) == sort([
        :posterior,
        :posterior_predictive,
        :log_likelihood,
        :sample_stats,
        :prior,
        :prior_predictive,
        :sample_stats_prior,
        :observed_data,
        :constant_data,
    ])
    @test length(idata5.prior.chain.values) == 3
    @test length(idata5.prior.draw.values) == 100
    @test idata5.observed_data.xs.values == xs
    @test only(idata5.observed_data.y.values) == y
    @test only(idata5.constant_data.n.values) == 5

    rng1 = Random.MersenneTwister(42)
    idata6 = from_turing(chn; model=model, rng=rng1)
    rng2 = Random.MersenneTwister(42)
    idata7 = from_turing(chn; model=model, rng=rng2)
    @testset for name in groupnames(idata6)
        group1 = getproperty(idata6, name)
        group2 = getproperty(idata7, name)
        @testset for var_name in group1.variables.keys()
            @test group1[var_name].values == group2[var_name].values
        end
    end
end
