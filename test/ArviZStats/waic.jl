using Test
using ArviZ
using ArviZ.ArviZStats
using ArviZExampleData
using DimensionalData
using Logging: SimpleLogger, with_logger

include("helpers.jl")

@testset "waic" begin
    @testset "core functionality" begin
        @testset for sz in ((1000, 4), (1000, 4, 2), (100, 4, 2, 3)),
            T in (Float32, Float64),
            TA in (Array, DimArray)

            atol_perm = cbrt(eps(T))

            log_likelihood = randn(T, sz)
            if TA === DimArray
                log_likelihood = DimArray(
                    log_likelihood, (:draw, :chain, :param1, :param2)[1:length(sz)]
                )
            end
            waic_result =
                TA === DimArray ? waic(log_likelihood) : @inferred(waic(log_likelihood))
            @test waic_result isa ArviZStats.WAICResult
            estimates = elpd_estimates(waic_result)
            pointwise = elpd_estimates(waic_result; pointwise=true)
            @testset "return types and values as expected" begin
                @test estimates isa NamedTuple{(:elpd, :elpd_mcse, :p, :p_mcse),NTuple{4,T}}
                @test pointwise isa NamedTuple{(:elpd, :p)}
                if length(sz) == 2
                    @test eltype(pointwise) === T
                else
                    @test eltype(pointwise) <: TA{T,length(sz) - 2}
                end
            end
            @testset "information criterion" begin
                @test information_criterion(waic_result, :log) == estimates.elpd
                @test information_criterion(waic_result, :negative_log) == -estimates.elpd
                @test information_criterion(waic_result, :deviance) == -2 * estimates.elpd
                @test information_criterion(waic_result, :log; pointwise=true) ==
                    pointwise.elpd
                @test information_criterion(waic_result, :negative_log; pointwise=true) ==
                    -pointwise.elpd
                @test information_criterion(waic_result, :deviance; pointwise=true) ==
                    -2 * pointwise.elpd
            end

            TA === DimArray && @testset "consistency with InferenceData argument" begin
                idata1 = InferenceData(; log_likelihood=Dataset((; x=log_likelihood)))
                waic_result1 = waic(idata1)
                @test isequal(waic_result1.estimates, waic_result.estimates)
                @test waic_result1.pointwise isa Dataset
                if length(sz) == 2
                    @test issetequal(keys(waic_result1.pointwise), (:elpd, :p))
                else
                    @test waic_result1.pointwise.elpd == waic_result.pointwise.elpd
                    @test waic_result1.pointwise.p == waic_result.pointwise.p
                end

                ll_perm = permutedims(
                    log_likelihood, (ntuple(x -> x + 2, length(sz) - 2)..., 2, 1)
                )
                idata2 = InferenceData(; log_likelihood=Dataset((; y=ll_perm)))
                waic_result2 = waic(idata2)
                @test waic_result2.estimates.elpd ≈ waic_result1.estimates.elpd atol =
                    atol_perm
                @test isapprox(
                    waic_result2.estimates.elpd_mcse,
                    waic_result1.estimates.elpd_mcse;
                    nans=true,
                    atol=atol_perm,
                )
                @test waic_result2.estimates.p ≈ waic_result1.estimates.p atol =
                    atol_perm
                @test isapprox(
                    waic_result2.estimates.p_mcse,
                    waic_result1.estimates.p_mcse;
                    nans=true,
                    atol=atol_perm,
                )
                @test waic_result2.pointwise.p ≈ waic_result1.pointwise.p atol =
                    atol_perm
            end
        end
    end
    @testset "warnings" begin
        io = IOBuffer()
        log_likelihood = randn(100, 4)
        @testset for bad_val in (NaN, -Inf, Inf)
            log_likelihood[1] = bad_val
            result = with_logger(SimpleLogger(io)) do
                waic(log_likelihood)
            end
            msg = String(take!(io))
            @test occursin("Warning:", msg)
        end
    end
    @testset "show" begin
        idata = load_example_data("centered_eight")
        # regression test
        @test sprint(show, "text/plain", waic(idata)) == """
            WAICResult with estimates
                   Estimate    SE
             elpd       -31   1.4
                p       0.9  0.33"""
    end
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
            @warn "Skipping consistency tests against R loo::waic, since loo is not installed."
            @test_broken false
        end
    end
end
