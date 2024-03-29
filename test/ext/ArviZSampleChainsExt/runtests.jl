using ArviZ
using SampleChains: SampleChains
using SampleChains.TupleVectors: TupleVectors
using Test

include("../../helpers.jl")

# minimal AbstractChain implementation
struct TestChain{T} <: SampleChains.AbstractChain{T}
    samples::TupleVectors.TupleVector{T}
    info::AbstractVector
end
SampleChains.samples(chain::TestChain) = getfield(chain, :samples)
SampleChains.info(chain::TestChain) = getfield(chain, :info)

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
end
