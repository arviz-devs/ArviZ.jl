using ArviZ
using SampleChains: SampleChains
using SampleChains.TupleVectors: TupleVectors
using SampleChainsDynamicHMC: SampleChainsDynamicHMC
using SampleChainsDynamicHMC.TransformVariables
using Test

include("../../helpers.jl")

function samplechains_dynamichmc_sample(nchains, ndraws)
    # μ ~ Normal(0, 1) |> iid(2, 3), σ ~ HalfNormal(0, 1), y[i] ~ Normal(μ[i], σ)
    y = [0.74 0.15 -1.08; -0.42 1.08 -0.52]
    function ℓ(nt)
        μ = nt.μ
        σ = nt.σ
        return -(sum(abs2, μ) + σ^2 + sum((y .- μ) .^ 2) / σ^2) / 2 - length(y) * log(σ)
    end
    t = as((μ=as(Array, 2, 3), σ=asℝ₊))
    chain = SampleChains.newchain(nchains, SampleChainsDynamicHMC.dynamichmc(), ℓ, t)
    return SampleChains.sample!(chain, ndraws - 1)
end

@testset "SampleChainsDynamicHMC" begin
    expected_stats_vars = (
        :acceptance_rate, :n_steps, :diverging, :energy, :tree_depth, :turning
    )

    multichain = samplechains_dynamichmc_sample(4, 10)
    idata = convert_to_inference_data(multichain)
    @test InferenceObjects.groupnames(idata) === (:posterior, :sample_stats)
    @test issubset(expected_stats_vars, keys(idata.sample_stats))
    @test size(idata.posterior.μ) == (10, 4, 2, 3)
    @test size(idata.posterior.σ) == (10, 4)

    idata = convert_to_inference_data(multichain; group=:prior)
    @test InferenceObjects.groupnames(idata) === (:prior, :sample_stats_prior)
    @test issubset(expected_stats_vars, keys(idata.sample_stats_prior))
end
