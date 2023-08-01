using ArviZ
using ArviZExampleData
using DimensionalData
using StatsBase
using Test

_maybescalar(x) = x
_maybescalar(x::AbstractArray{<:Any,0}) = x[]

_maybevec(x::AbstractArray) = vec(x)
_maybevec(x) = x

@testset "summary statistics" begin
    @testset "summarystats" begin
        @testset "return_type=Dataset" begin
            data = (x=randn(100, 2), y=randn(100, 2, 3), z=randn(100, 2, 2, 3))
            ds = convert_to_dataset(
                data;
                dims=(y=[:a], z=[:b, :c]),
                coords=(a=["q", "r", "s"], b=[0, 1], c=[:m, :n, :o]),
            )

            stats = @inferred Dataset summarystats(ds; return_type=Dataset)
            @test issetequal(keys(stats), keys(ds))
            @test isempty(DimensionalData.metadata(stats))
            @test hasdim(stats, :_metric)
            @test lookup(stats, :_metric) == [
                "mean",
                "std",
                "hdi_3%",
                "hdi_97%",
                "mcse_mean",
                "mcse_std",
                "ess_tail",
                "ess_bulk",
                "rhat",
            ]
            @test view(stats; _metric=At("mean")) ==
                dropdims(mean(ds; dims=(:draw, :chain)); dims=(:draw, :chain))
            @test view(stats; _metric=At("std")) ==
                dropdims(std(ds; dims=(:draw, :chain)); dims=(:draw, :chain))
            hdi_ds = hdi(ds)
            @test stats[_metric=At("hdi_3%")] == hdi_ds[hdi_bound=1]
            @test stats[_metric=At("hdi_97%")] == hdi_ds[hdi_bound=2]
            @test view(stats; _metric=At("mcse_mean")) == mcse(ds; kind=mean)
            @test view(stats; _metric=At("mcse_std")) == mcse(ds; kind=std)
            @test view(stats; _metric=At("ess_tail")) == ess(ds; kind=:tail)
            @test view(stats; _metric=At("ess_bulk")) == ess(ds; kind=:bulk)
            @test view(stats; _metric=At("rhat")) == rhat(ds)

            stats2 = summarystats(ds; return_type=Dataset, prob_interval=0.8)
            @test lookup(stats2, :_metric) == [
                "mean",
                "std",
                "hdi_10%",
                "hdi_90%",
                "mcse_mean",
                "mcse_std",
                "ess_tail",
                "ess_bulk",
                "rhat",
            ]
            hdi_ds2 = hdi(ds; prob=0.8)
            @test stats2[_metric=At("hdi_10%")] == hdi_ds2[hdi_bound=1]
            @test stats2[_metric=At("hdi_90%")] == hdi_ds2[hdi_bound=2]

            stats3 = summarystats(ds; return_type=Dataset, kind=:stats)
            @test lookup(stats3, :_metric) == ["mean", "std", "hdi_3%", "hdi_97%"]
            @test stats3 == stats[_metric=At(["mean", "std", "hdi_3%", "hdi_97%"])]

            stats4 = summarystats(ds; return_type=Dataset, kind=:diagnostics)
            @test lookup(stats4, :_metric) ==
                ["mcse_mean", "mcse_std", "ess_tail", "ess_bulk", "rhat"]
            @test stats4 == stats[_metric=At([
                "mcse_mean", "mcse_std", "ess_tail", "ess_bulk", "rhat"
            ])]

            stats5 = summarystats(
                ds;
                return_type=Dataset,
                metric_dim=:__metric,
                kind=:stats,
                prob_interval=0.95,
            )
            @test lookup(stats5, :__metric) == ["mean", "std", "hdi_2.5%", "hdi_97.5%"]
        end

        @testset "_indices_iterator" begin
            data = (x=randn(3), y=randn(3, 3), z=randn(3, 2, 3))
            ds = namedtuple_to_dataset(
                data;
                dims=(x=[:s], y=[:s, :a], z=[:s, :b, :c]),
                coords=(a=["q", "r", "s"], b=[0, 1], c=[:m, :n, :o]),
                default_dims=(),
            )
            var_inds = collect(ArviZStats._indices_iterator(ds, :s))
            @test length(var_inds) == 1 + size(ds.y, :a) + size(ds.z, :b) * size(ds.z, :c)
            @test var_inds[1] == (data.x, ())
            @test first.(var_inds[2:4]) == fill(data.y, size(ds.y, :a))
            @test last.(var_inds[2:4]) == vec(DimensionalData.DimKeys(otherdims(ds.y, :s)))
            @test first.(var_inds[5:end]) == fill(data.z, size(ds.z, :b) * size(ds.z, :c))
            @test last.(var_inds[5:end]) ==
                vec(DimensionalData.DimKeys(otherdims(ds.z, :s)))
        end

        @testset "_indices_to_name" begin
            data = (x=randn(3), y=randn(3, 3), z=randn(3, 2, 3))
            ds = namedtuple_to_dataset(
                data;
                dims=(x=[:s], y=[:s, :a], z=[:s, :b, :c]),
                coords=(a=["q", "r", "s"], b=[0, 1], c=[:m, :n, :o]),
                default_dims=(),
            )

            @test ArviZStats._indices_to_name(ds.x, (), true) == "x"
            @test ArviZStats._indices_to_name(ds.x, (), false) == "x"
            y_names = map(DimensionalData.DimKeys(dims(ds.y, :a))) do d
                return ArviZStats._indices_to_name(ds.y, d, true)
            end
            @test y_names == ["y[q]", "y[r]", "y[s]"]
            y_names_verbose = map(DimensionalData.DimKeys(dims(ds.y, :a))) do d
                return ArviZStats._indices_to_name(ds.y, d, false)
            end
            @test y_names_verbose == ["y[a=At(\"q\")]", "y[a=At(\"r\")]", "y[a=At(\"s\")]"]
            z_names = vec(
                map(DimensionalData.DimKeys(dims(ds.z, (:b, :c)))) do d
                    return ArviZStats._indices_to_name(ds.z, d, true)
                end,
            )
            @test z_names == ["z[0,m]", "z[1,m]", "z[0,n]", "z[1,n]", "z[0,o]", "z[1,o]"]
            z_names_verbose = vec(
                map(DimensionalData.DimKeys(dims(ds.z, (:b, :c)))) do d
                    return ArviZStats._indices_to_name(ds.z, d, false)
                end,
            )
            @test z_names_verbose == [
                "z[b=At(0),c=At(:m)]",
                "z[b=At(1),c=At(:m)]",
                "z[b=At(0),c=At(:n)]",
                "z[b=At(1),c=At(:n)]",
                "z[b=At(0),c=At(:o)]",
                "z[b=At(1),c=At(:o)]",
            ]
        end

        @testset "return_type=SummaryStats" begin
            data = (x=randn(100, 2), y=randn(100, 2, 3), z=randn(100, 2, 2, 3))
            ds = convert_to_dataset(
                data;
                dims=(y=[:a], z=[:b, :c]),
                coords=(a=["q", "r", "s"], b=[0, 1], c=[:m, :n, :o]),
            )

            stats = @inferred SummaryStats summarystats(ds)
            stats_data = parent(stats)
            @test stats_data.variable == [
                "x",
                "y[q]",
                "y[r]",
                "y[s]",
                "z[0,m]",
                "z[1,m]",
                "z[0,n]",
                "z[1,n]",
                "z[0,o]",
                "z[1,o]",
            ]
            @test issetequal(
                keys(stats_data),
                [
                    :variable,
                    :mean,
                    :std,
                    Symbol("hdi_3%"),
                    Symbol("hdi_97%"),
                    :mcse_mean,
                    :mcse_std,
                    :ess_tail,
                    :ess_bulk,
                    :rhat,
                ],
            )

            # check a handful of values
            @test stats_data.mean == reduce(
                vcat, map(vec, DimensionalData.layers(mean(ds; dims=(:draw, :chain))))
            )
            @test stats_data.std == reduce(
                vcat, map(vec, DimensionalData.layers(std(ds; dims=(:draw, :chain))))
            )
            hdi_ds = hdi(ds)
            @test stats_data[Symbol("hdi_3%")] ==
                reduce(vcat, map(_maybevec, DimensionalData.layers(hdi_ds[hdi_bound=1])))
            @test stats_data[Symbol("hdi_97%")] ==
                reduce(vcat, map(_maybevec, DimensionalData.layers(hdi_ds[hdi_bound=2])))
            @test stats_data.mcse_mean ==
                reduce(vcat, map(_maybevec, DimensionalData.layers(mcse(ds; kind=mean))))
            @test stats_data.mcse_std ==
                reduce(vcat, map(_maybevec, DimensionalData.layers(mcse(ds; kind=std))))
            @test stats_data.ess_bulk ==
                reduce(vcat, map(_maybevec, DimensionalData.layers(ess(ds))))
            @test stats_data.rhat ==
                reduce(vcat, map(_maybevec, DimensionalData.layers(rhat(ds))))

            stats2 = summarystats(ds; prob_interval=0.8, kind=:stats)
            @test issetequal(
                keys(parent(stats2)),
                [:variable, :mean, :std, Symbol("hdi_10%"), Symbol("hdi_90%")],
            )
        end
    end
end
