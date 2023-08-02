using Test
using ArviZ
using ArviZ.ArviZStats
using ArviZExampleData
using DimensionalData
using Logging: SimpleLogger, with_logger

include("helpers.jl")

@testset "loo" begin
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
            loo_result =
                TA === DimArray ? loo(log_likelihood) : @inferred(loo(log_likelihood))
            @test loo_result isa ArviZStats.PSISLOOResult
            estimates = elpd_estimates(loo_result)
            pointwise = elpd_estimates(loo_result; pointwise=true)
            @testset "return types and values as expected" begin
                @test estimates isa NamedTuple{(:elpd, :elpd_mcse, :p, :p_mcse),NTuple{4,T}}
                @test pointwise isa
                    NamedTuple{(:elpd, :elpd_mcse, :p, :reff, :pareto_shape)}
                if length(sz) == 2
                    @test eltype(pointwise) === T
                else
                    @test eltype(pointwise) <: TA{T,length(sz) - 2}
                end
                @test loo_result.psis_result isa PSIS.PSISResult
                @test loo_result.psis_result.reff == pointwise.reff
                @test loo_result.psis_result.pareto_shape == pointwise.pareto_shape
            end
            @testset "information criterion" begin
                @test information_criterion(loo_result, :log) == estimates.elpd
                @test information_criterion(loo_result, :negative_log) == -estimates.elpd
                @test information_criterion(loo_result, :deviance) == -2 * estimates.elpd
                @test information_criterion(loo_result, :log; pointwise=true) ==
                    pointwise.elpd
                @test information_criterion(loo_result, :negative_log; pointwise=true) ==
                    -pointwise.elpd
                @test information_criterion(loo_result, :deviance; pointwise=true) ==
                    -2 * pointwise.elpd
            end

            TA === DimArray && @testset "consistency with InferenceData argument" begin
                idata1 = InferenceData(; log_likelihood=Dataset((; x=log_likelihood)))
                loo_result1 = loo(idata1)
                @test isequal(loo_result1.estimates, loo_result.estimates)
                @test loo_result1.pointwise isa Dataset
                if length(sz) == 2
                    @test issetequal(
                        keys(loo_result1.pointwise),
                        (:elpd, :elpd_mcse, :p, :reff, :pareto_shape),
                    )
                else
                    @test loo_result1.pointwise.elpd == loo_result.pointwise.elpd
                    @test loo_result1.pointwise.elpd_mcse == loo_result.pointwise.elpd_mcse
                    @test loo_result1.pointwise.p == loo_result.pointwise.p
                    @test loo_result1.pointwise.reff == loo_result.pointwise.reff
                    @test loo_result1.pointwise.pareto_shape ==
                        loo_result.pointwise.pareto_shape
                end

                ll_perm = permutedims(
                    log_likelihood, (ntuple(x -> x + 2, length(sz) - 2)..., 2, 1)
                )
                idata2 = InferenceData(; log_likelihood=Dataset((; y=ll_perm)))
                loo_result2 = loo(idata2)
                @test loo_result2.estimates.elpd ≈ loo_result1.estimates.elpd atol =
                    atol_perm
                @test isapprox(
                    loo_result2.estimates.elpd_mcse,
                    loo_result1.estimates.elpd_mcse;
                    nans=true,
                    atol=atol_perm,
                )
                @test loo_result2.estimates.p ≈ loo_result1.estimates.p atol = atol_perm
                @test isapprox(
                    loo_result2.estimates.p_mcse,
                    loo_result1.estimates.p_mcse;
                    nans=true,
                    atol=atol_perm,
                )
                @test isapprox(
                    loo_result2.pointwise.elpd_mcse,
                    loo_result1.pointwise.elpd_mcse;
                    nans=true,
                    atol=atol_perm,
                )
                @test loo_result2.pointwise.p ≈ loo_result1.pointwise.p atol = atol_perm
                @test loo_result2.pointwise.reff ≈ loo_result1.pointwise.reff atol =
                    atol_perm
                @test loo_result2.pointwise.pareto_shape ≈
                    loo_result1.pointwise.pareto_shape atol = atol_perm
            end
        end
    end
    @testset "keywords forwarded" begin
        log_likelihood = convert_to_dataset((x=randn(1000, 4, 2, 3), y=randn(1000, 4, 3)))
        @test loo(log_likelihood; var_name=:x).estimates == loo(log_likelihood.x).estimates
        @test loo(log_likelihood; var_name=:y).estimates == loo(log_likelihood.y).estimates
        @test loo(log_likelihood; var_name=:x, reff=0.5).pointwise.reff == fill(0.5, 2, 3)
    end
    @testset "errors" begin
        log_likelihood = convert_to_dataset((x=randn(1000, 4, 2, 3), y=randn(1000, 4, 3)))
        @test_throws ArgumentError loo(log_likelihood)
        @test_throws ArgumentError loo(log_likelihood; var_name=:z)
        @test_throws DimensionMismatch loo(log_likelihood; var_name=:x, reff=rand(2))
    end
    @testset "warnings" begin
        io = IOBuffer()
        log_likelihood = randn(100, 4)
        @testset for bad_val in (NaN, -Inf, Inf)
            log_likelihood[1] = bad_val
            result = with_logger(SimpleLogger(io)) do
                loo(log_likelihood)
            end
            msg = String(take!(io))
            @test occursin("Warning:", msg)
        end

        io = IOBuffer()
        log_likelihood = randn(100, 4)
        @testset for bad_reff in (NaN, 0, Inf)
            result = with_logger(SimpleLogger(io)) do
                loo(log_likelihood; reff=bad_reff)
            end
            msg = String(take!(io))
            @test occursin("Warning:", msg)
        end

        io = IOBuffer()
        log_likelihood = randn(5, 1)
        result = with_logger(SimpleLogger(io)) do
            loo(log_likelihood)
        end
        msg = String(take!(io))
        @test occursin("Warning:", msg)
    end
    @testset "show" begin
        idata = load_example_data("centered_eight")
        # regression test
        @test sprint(show, "text/plain", loo(idata)) == """
            PSISLOOResult with estimates
             elpd  elpd_mcse    p  p_mcse
              -31        1.4  0.9    0.34

            and PSISResult with 500 draws, 4 chains, and 8 parameters
            Pareto shape (k) diagnostic values:
                                Count      Min. ESS
             (-Inf, 0.5]  good  6 (75.0%)  135
              (0.5, 0.7]  okay  2 (25.0%)  421"""
    end
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
            @warn "Skipping consistency tests against R loo::loo, since loo is not installed."
            @test_broken false
        end
    end
end
