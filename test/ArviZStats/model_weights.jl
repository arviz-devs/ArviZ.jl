using Test
using ArviZ
using DimensionalData
using Optim
using Random

Random.seed!(97)

@testset "model_weights" begin
    function test_model_weights(weights_method)
        @testset "weights are same collection as arguments" begin
            elpd_results_tuple = map(loo, (randn(1000, 4, 2, 3), randn(1000, 4, 2, 3)))
            weights_tuple = @inferred model_weights(weights_method(), elpd_results_tuple)
            @test weights_tuple isa NTuple{2,Float64}
            @test sum(weights_tuple) ≈ 1

            elpd_results_nt = NamedTuple{(:x, :y)}(elpd_results_tuple)
            weights_nt = @inferred model_weights(weights_method(), elpd_results_nt)
            @test weights_nt isa NamedTuple{(:x, :y),NTuple{2,Float64}}
            @test _isapprox(values(weights_nt), weights_tuple)

            elpd_results_da = DimArray(collect(elpd_results_tuple), Dim{:model}([:x, :y]))
            weights_da = @inferred model_weights(weights_method(), elpd_results_da)
            @test weights_da isa DimArray
            @test Dimensions.dimsmatch(weights_da, elpd_results_da)
            @test _isapprox(weights_da, collect(weights_tuple))
        end

        @testset "weights invariant to order" begin
            elpd_results = map(
                waic, (randn(1000, 4, 10), randn(1000, 4, 10), randn(1000, 4, 10))
            )
            weights1 = model_weights(weights_method(), elpd_results)
            weights2 = model_weights(weights_method(), reverse(elpd_results))
            @test _isapprox(weights1, reverse(weights2))
        end

        @testset "identical models get the same weights" begin
            ll = randn(1000, 4, 10)
            result = waic(ll)
            elpd_results = fill(result, 3)
            weights = model_weights(weights_method(), elpd_results)
            @test sum(weights) ≈ 1
            @test weights ≈ fill(weights[1], length(weights))
        end

        @testset "better model gets higher weight" begin
            elpd_results = (
                non_centered=loo(load_example_data("non_centered_eight")),
                centered=loo(load_example_data("centered_eight")),
            )
            weights = model_weights(weights_method(), elpd_results)
            @test sum(weights) ≈ 1
            @test weights[1] > weights[2]
        end
    end

    @testset "PseudoBMA" begin
        @test !PseudoBMA().regularize
        @test PseudoBMA(true) === PseudoBMA(; regularize=true)

        test_model_weights(PseudoBMA)

        @testset "regularization is respected" begin
            elpd_results = map(waic, [randn(1000, 4, 2, 3) for _ in 1:2])
            weights_reg = model_weights(PseudoBMA(true), elpd_results)
            weights_nonreg = model_weights(PseudoBMA(false), elpd_results)
            @test !(weights_reg ≈ weights_nonreg)
        end
    end
    @testset "BootstrappedPseudoBMA" begin
        test_model_weights() do
            # use the same seed for every run
            rng = MersenneTwister(37)
            BootstrappedPseudoBMA(; rng)
        end

        @testset "number of samples can be configured" begin
            elpd_results = map(waic, [randn(1000, 4, 2, 3) for _ in 1:2])
            rng = MersenneTwister(64)
            weights1 = model_weights(BootstrappedPseudoBMA(; rng, samples=10), elpd_results)
            rng = MersenneTwister(64)
            weights2 = model_weights(
                BootstrappedPseudoBMA(; rng, samples=100), elpd_results
            )
            @test !(weights1 ≈ weights2)
        end
    end
    @testset "Stacking" begin
        @testset "stacking is default" begin
            elpd_results = map(waic, [randn(1000, 4, 2, 3) for _ in 1:2])
            @test model_weights(elpd_results) == model_weights(Stacking(), elpd_results)
        end

        test_model_weights(Stacking)

        @testset "alternate optimizer options are used" begin
            elpd_results = map(waic, [randn(1000, 4, 2, 3) for _ in 1:10])
            weights1 = model_weights(Stacking(), elpd_results)
            weights2 = model_weights(Stacking(), elpd_results)
            optimizer = GradientDescent()
            weights3 = model_weights(Stacking(; optimizer), elpd_results)
            options = Optim.Options(; iterations=2)
            weights4 = model_weights(Stacking(; options), elpd_results)
            @test weights3 != weights1 == weights2 != weights4
            @test weights3 ≈ weights1
        end
    end
end
