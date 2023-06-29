using Test
using ArviZ
using ArviZ.ArviZStats
using DimensionalData

include("helpers.jl")

@testset "loo" begin
    @testset "agrees with R loo" begin
        if r_loo_installed()
            @testset for ds_name in ["centered_eight", "non_centered_eight"]
                idata = load_example_data(ds_name)
                log_likelihood = idata.log_likelihood.obs
                data_dims = otherdims(log_likelihood, (:draw, :chain))
                log_likelihood = permutedims(log_likelihood, (:draw, :chain, data_dims...))
                reff_rand = rand(data_dims)
                @testset for reff in (nothing, reff_rand)
                    result_r = loo_r(log_likelihood; reff)
                    result = loo(log_likelihood; reff)
                    @test result.estimates.elpd ≈ result_r.estimates.elpd
                    @test result.estimates.elpd_mcse ≈ result_r.estimates.elpd_mcse
                    @test result.estimates.p ≈ result_r.estimates.p
                    @test result.estimates.p_mcse ≈ result_r.estimates.p_mcse
                    @test result.pointwise.elpd ≈ result_r.pointwise.elpd
                    # increased tolerance for elpd_mcse, since we use a different approach
                    @test result.pointwise.elpd_mcse ≈ result_r.pointwise.elpd_mcse rtol =
                        0.01
                    @test result.pointwise.p ≈ result_r.pointwise.p
                    @test result.pointwise.reff ≈ result_r.pointwise.reff
                    @test result.pointwise.pareto_shape ≈ result_r.pointwise.pareto_shape
                end
            end
        else
            @test_broken false
        end
    end
end
