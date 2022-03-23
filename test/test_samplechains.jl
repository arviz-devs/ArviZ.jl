using SampleChains: SampleChains
using SampleChains.TupleVectors: TupleVectors
using SampleChainsDynamicHMC: SampleChainsDynamicHMC
using SampleChainsDynamicHMC.TransformVariables

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
        dims = Dict(:y => [:a], :z => [:b, :c])
        coords = Dict(:a => ["a1", "a2"], :b => ["b1", "b2"], :c => ["c1", "c2", "c3"])
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
                map((idata, idata_conv)) do _idata
                    test_namedtuple_data(
                        _idata, group, propertynames(multichain), nchains, ndraws; kwargs...
                    )
                    @test only(ArviZ.groupnames(_idata)) === group
                end
                @testset for var_name in propertynames(multichain)
                    @test getproperty(getproperty(idata, group), var_name).values ==
                        getproperty(getproperty(idata_nt, group), var_name).values
                    @test getproperty(getproperty(idata, group), var_name).values ==
                        getproperty(getproperty(idata_conv, group), var_name).values
                end
            end
        end
    end
    @testset "SampleChainsDynamicHMC" begin
        expected_stats_vars = (
            :acceptance_rate, :n_steps, :diverging, :energy, :tree_depth, :turning
        )

        multichain = samplechains_dynamichmc_sample(4, 10)
        idata = convert_to_inference_data(multichain)
        @test sort([:posterior, :sample_stats]) == ArviZ.groupnames(idata)
        stats_vars = setdiff(map(Symbol, idata.sample_stats.variables), (:chain, :draw))
        missing_vars = setdiff(expected_stats_vars, stats_vars)
        @test isempty(missing_vars)
        post_vars = setdiff(map(Symbol, idata.posterior.variables), (:chain, :draw))
        @test issubset((:μ, :σ), post_vars)
        @test size(idata.posterior.μ.values) == (4, 10, 2, 3)
        @test size(idata.posterior.σ.values) == (4, 10)

        idata = convert_to_inference_data(multichain; group=:prior)
        @test sort([:prior, :sample_stats_prior]) == ArviZ.groupnames(idata)
        stats_vars = setdiff(
            map(Symbol, idata.sample_stats_prior.variables), (:chain, :draw)
        )
        missing_vars = setdiff(expected_stats_vars, stats_vars)
        @test isempty(missing_vars)
    end
end
