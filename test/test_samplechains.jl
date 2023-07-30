using ArviZ
using DimensionalData
using SampleChains: SampleChains
using SampleChains.TupleVectors: TupleVectors
using SampleChainsDynamicHMC: SampleChainsDynamicHMC
using SampleChainsDynamicHMC.TransformVariables
using Test

# minimal AbstractChain implementation
struct TestChain{T} <: SampleChains.AbstractChain{T}
    samples::TupleVectors.TupleVector{T}
    info::AbstractVector
end
SampleChains.samples(chain::TestChain) = getfield(chain, :samples)
SampleChains.info(chain::TestChain) = getfield(chain, :info)

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

@testset "SampleChains" begin
    @testset "TestChain with $nchains chains" for nchains in (1, 4)
        ndraws = 10
        dims = (y=[:a], z=[:b, :c])
        coords = (a=["a1", "a2"], b=["b1", "b2"], c=["c1", "c2", "c3"])
        tvs = map(1:nchains) do _
            init = (x=randn(), y=randn(2), z=randn(2, 3))
            tv = TupleVectors.TupleVector(undef, init, ndraws)
            copyto!(tv.x, randn(ndraws))
            copyto!(tv.y, [randn(2) for _ in 1:ndraws])
            copyto!(tv.z, [randn(2, 3) for _ in 1:ndraws])
            return tv
        end
        nts = map(collect, tvs)
        info = [(x=3, y=4) for _ in 1:ndraws]
        chains = [TestChain(tv, info) for tv in tvs]
        multichain = SampleChains.MultiChain(chains)

        kwargs = (dims=dims, coords=coords, library="MyLib")
        data = Dict(
            "Vector{AbstractChain}" => chains,
            "NTuple{N,AbstractChain}" => Tuple(chains),
            "MultiChain" => multichain,
        )
        if nchains === 1
            data["AbstractChain"] = only(chains)
        end

        @testset "$k" for (k, chaindata) in data
            @testset "$group" for group in (:posterior, :prior)
                if group === :posterior
                    idata = from_samplechains(chaindata; kwargs...)
                    idata_nt = from_namedtuple(nts; kwargs...)
                else
                    idata = from_samplechains(; group => chaindata, kwargs...)
                    idata_nt = from_namedtuple(; group => nts, kwargs...)
                end
                idata_conv = convert_to_inference_data(chaindata; group, kwargs...)
                test_idata_approx_equal(idata, idata_nt)
                test_idata_approx_equal(idata, idata_conv)
            end
        end
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
end
