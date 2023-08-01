using Test
using ArviZ
using DimensionalData
using IteratorInterfaceExtensions
using Tables
using TableTraits

function _isequal(x::ModelComparisonResult, y::ModelComparisonResult)
    return Tables.columntable(x) == Tables.columntable(y)
end

@testset "compare" begin
    eight_schools_data = (
        centered=load_example_data("centered_eight"),
        non_centered=load_example_data("non_centered_eight"),
    )
    eight_schools_loo_results = map(loo, eight_schools_data)
    mc1 = @inferred ModelComparisonResult compare(eight_schools_loo_results)

    @testset "basic checks" begin
        @test mc1.name == (:non_centered, :centered)
        @test mc1.rank == (non_centered=1, centered=2)
        @test _isapprox(
            mc1.elpd_diff,
            (
                non_centered=0.0,
                centered=(
                    eight_schools_loo_results.non_centered.estimates.elpd -
                    eight_schools_loo_results.centered.estimates.elpd
                ),
            ),
        )
        @test mc1.elpd_diff.non_centered == 0.0
        @test mc1.elpd_diff.centered > 0
        @test mc1.weight == NamedTuple{(:non_centered, :centered)}(
            model_weights(eight_schools_loo_results)
        )
        @test mc1.elpd_result ==
            NamedTuple{(:non_centered, :centered)}(eight_schools_loo_results)

        @test_throws ArgumentError compare(eight_schools_loo_results; model_names=[:foo])
        @test_throws ErrorException compare(eight_schools_data; elpd_method=x -> nothing)
    end

    @testset "keywords are forwarded" begin
        @test _isequal(compare(eight_schools_data), mc1)
        mc2 = compare(eight_schools_loo_results; weights_method=PseudoBMA())
        @test !_isequal(mc2, compare(eight_schools_loo_results))
        @test mc2.weights_method === PseudoBMA()
        mc3 = compare(eight_schools_loo_results; sort=false)
        for k in filter(!=(:weights_method), propertynames(mc1))
            if k === :name
                @test getproperty(mc3, k) == reverse(getproperty(mc1, k))
            else
                @test getproperty(mc3, k) ==
                    NamedTuple{(:centered, :non_centered)}(getproperty(mc1, k))
            end
        end
        mc3 = compare(eight_schools_loo_results; model_names=[:a, :b])
        @test mc3.name == [:b, :a]
        mc4 = compare(eight_schools_data; elpd_method=waic)
        @test !_isequal(mc4, mc2)
    end

    @testset "ModelComparisonResult" begin
        @testset "Tables interface" begin
            @test Tables.istable(typeof(mc1))
            @test Tables.columnaccess(typeof(mc1))
            @test Tables.columns(mc1) == mc1
            @test Tables.columnnames(mc1) == (
                :name,
                :rank,
                :elpd,
                :elpd_mcse,
                :elpd_diff,
                :elpd_diff_mcse,
                :weight,
                :p,
                :p_mcse,
            )
            table = Tables.columntable(mc1)
            for k in (:name, :rank, :elpd_diff, :elpd_diff_mcse, :weight)
                @test getproperty(table, k) == collect(getproperty(mc1, k))
            end
            for k in (:elpd, :elpd_mcse, :p, :p_mcse)
                @test getproperty(table, k) ==
                    collect(map(x -> getproperty(x.estimates, k), mc1.elpd_result))
            end
            for (i, k) in enumerate(Tables.columnnames(mc1))
                @test Tables.getcolumn(mc1, i) == Tables.getcolumn(mc1, k)
            end
            @test_throws ArgumentError Tables.getcolumn(mc1, :foo)
        end

        @testset "TableTraits interface" begin
            @test IteratorInterfaceExtensions.isiterable(mc1)
            @test TableTraits.isiterabletable(mc1)
            nt = collect(Iterators.take(IteratorInterfaceExtensions.getiterator(mc1), 1))[1]
            @test isequal(
                nt,
                (; (k => Tables.getcolumn(mc1, k)[1] for k in Tables.columnnames(mc1))...),
            )
            nt = collect(Iterators.take(IteratorInterfaceExtensions.getiterator(mc1), 2))[2]
            @test isequal(
                nt,
                (; (k => Tables.getcolumn(mc1, k)[2] for k in Tables.columnnames(mc1))...),
            )
        end

        @testset "show" begin
            mc5 = compare(eight_schools_loo_results; weights_method=PseudoBMA())
            @test sprint(show, "text/plain", mc1) == """
                ModelComparisonResult with Stacking weights
                 name          rank  elpd  elpd_mcse  elpd_diff  elpd_diff_mcse  weight    p  p_mcse
                 non_centered     1   -31        1.4       0              0.0       1.0  0.9    0.32
                 centered         2   -31        1.4       0.06           0.067     0.0  0.9    0.34"""

            @test sprint(show, "text/plain", mc5) == """
                ModelComparisonResult with PseudoBMA weights
                 name          rank  elpd  elpd_mcse  elpd_diff  elpd_diff_mcse  weight    p  p_mcse
                 non_centered     1   -31        1.4       0              0.0      0.52  0.9    0.32
                 centered         2   -31        1.4       0.06           0.067    0.48  0.9    0.34"""
        end
    end
end
