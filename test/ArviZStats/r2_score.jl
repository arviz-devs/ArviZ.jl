using ArviZ
using ArviZExampleData
using GLM
using Statistics
using Test

@testset "r2_score/r2_sample" begin
    @testset "basic" begin
        n = 100
        @testset for T in (Float32, Float64),
            sz in (300, (100, 3)),
            σ in T.((2, 1, 0.5, 0.1))

            x = range(T(0), T(1); length=n)
            slope = T(2)
            intercept = T(3)
            y = @. slope * x + intercept + randn(T) * σ
            x_reshape = length(sz) == 1 ? x' : reshape(x, 1, 1, :)
            y_pred = slope .* x_reshape .+ intercept .+ randn(T, sz..., n) .* σ

            r2_val = @inferred r2_score(y, y_pred)
            @test r2_val isa NamedTuple{(:r2, :r2_std),NTuple{2,T}}
            r2_draws = @inferred ArviZStats.r2_samples(y, y_pred)
            @test r2_val.r2 ≈ mean(r2_draws)
            @test r2_val.r2_std ≈ std(r2_draws; corrected=false)

            # check rough consistency with GLM
            res = lm(@formula(y ~ 1 + x), (; x=Float64.(x), y=Float64.(y)))
            @test r2_val.r2 ≈ r2(res) rtol = 1
        end
    end

    @testset "InferenceData inputs" begin
        @testset for name in ("regression1d", "regression10d")
            idata = load_example_data(name)
            VERSION ≥ v"1.9" && @inferred r2_score(idata)
            r2_val = r2_score(idata)
            @test r2_val == r2_score(
                idata.observed_data.y,
                PermutedDimsArray(idata.posterior_predictive.y, (:draw, :chain, :y_dim_0)),
            )
            @test r2_val == r2_score(idata; y_name=:y)
            @test r2_val == r2_score(idata; y_pred_name=:y)
            @test r2_val == r2_score(idata; y_name=:y, y_pred_name=:y)
            @test_throws Exception r2_score(idata; y_name=:z)
            @test_throws Exception r2_score(idata; y_pred_name=:z)
        end
    end
end
