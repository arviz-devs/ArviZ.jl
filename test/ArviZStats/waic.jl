using Test
using ArviZ
using ArviZ.ArviZStats
using DimensionalData

include("helpers.jl")

@testset "waic" begin
    @testset "agrees with R waic" begin
        if r_loo_installed()
            @testset for ds_name in ["centered_eight", "non_centered_eight"]
                idata = load_example_data(ds_name)
                log_likelihood = idata.log_likelihood.obs
                data_dims = otherdims(log_likelihood, (:draw, :chain))
                log_likelihood = permutedims(log_likelihood, (:draw, :chain, data_dims...))
                reff_rand = rand(data_dims)
                result_r = waic_r(log_likelihood)
                result = waic(log_likelihood)
                @test result.estimates.elpd ≈ result_r.estimates.elpd
                @test result.estimates.elpd_mcse ≈ result_r.estimates.elpd_mcse
                @test result.estimates.p ≈ result_r.estimates.p
                @test result.estimates.p_mcse ≈ result_r.estimates.p_mcse
                @test result.pointwise.elpd ≈ result_r.pointwise.elpd
                @test result.pointwise.p ≈ result_r.pointwise.p
            end
        else
            @test_broken false
        end
    end
end
