using PyCall, PyPlot

@testset "plots" begin
    data = load_arviz_data("centered_eight")
    data2 = load_arviz_data("non_centered_eight")

    rng = MersenneTwister(42)
    arr1 = randn(rng, 4, 100)
    arr2 = randn(rng, 4, 100)
    arr3 = randn(rng, 100)

    @testset "$(f)" for f in (plot_trace, plot_pair, plot_joint)
        f(data; var_names = ["tau", "mu"])
        close(gcf())
        f((x = arr1, y = arr2); var_names = ["x", "y"])
        close(gcf())
        ispynull(ArviZ.bokeh) || @testset "bokeh" begin
            @test f((x = arr1, y = arr2); var_names = ["x", "y"], backend = :bokeh) isa
                  ArviZ.BokehPlot
        end
    end

    @testset "$(f)" for f in
                        (plot_autocorr, plot_ess, plot_mcse, plot_posterior, plot_violin)
        f(data; var_names = ["tau", "mu"])
        close(gcf())
        f(arr1)
        close(gcf())
        f((x = arr1, y = arr2); var_names = ["x", "y"])
        close(gcf())
        ispynull(ArviZ.bokeh) || @testset "bokeh" begin
            @test f((x = arr1, y = arr2); var_names = ["x", "y"], backend = :bokeh) isa
                  ArviZ.BokehPlot
        end
    end

    @testset "$(f)" for f in (plot_energy, plot_parallel)
        f(data)
        close(gcf())
        ispynull(ArviZ.bokeh) || @testset "bokeh" begin
            @test f(data; backend = :bokeh) isa ArviZ.BokehPlot
        end
    end

    @testset "$(f)" for f in (plot_density, plot_forest)
        f(data; var_names = ["tau", "mu"])
        close(gcf())
        f([(x = arr1,), (x = arr2,)], var_names = ["x"])
        close(gcf())
        f(arr3)
        close(gcf())
        f((x = arr1, y = arr2); var_names = ["x", "y"])
        close(gcf())
        ispynull(ArviZ.bokeh) || @testset "bokeh" begin
            @test f([(x = arr1,), (x = arr2,)]; var_names = ["x"], backend = :bokeh) isa
                  ArviZ.BokehPlot
        end
    end

    @testset "plot_rank" begin
        plot_rank(data; var_names = ["tau", "mu"])
        close(gcf())
        plot_rank(arr1)
        close(gcf())
        plot_rank((x = arr1, y = arr2); var_names = ["x", "y"])
        close(gcf())
        ispynull(ArviZ.bokeh) || @testset "bokeh" begin
            @test plot_rank(
                (x = arr1, y = arr2);
                var_names = ["x", "y"],
                backend = :bokeh,
            ) isa ArviZ.BokehPlot
        end
    end

    @testset "plot_compare" begin
        df = compare(Dict("a" => data, "b" => data2))
        plot_compare(df)
        close(gcf())
        ispynull(ArviZ.bokeh) || @testset "bokeh" begin
            @test plot_compare(df; backend = :bokeh) isa ArviZ.BokehPlot
        end
    end

    @testset "$(f)" for f in (plot_dist, plot_kde)
        f(arr1)
        close(gcf())
        ispynull(ArviZ.bokeh) || @testset "bokeh" begin
            @test f(arr1; backend = :bokeh) isa ArviZ.BokehPlot
        end
    end

    @testset "plot_hpd" begin
        x_data = randn(rng, 100)
        y_data = 2 .+ reshape(x_data, 1, 100) .* 0.5
        y_data_rep = 0.5 .* randn(rng, 200, 100) .* y_data
        plot_hpd(x_data, y_data_rep)
        close(gcf())
        ispynull(ArviZ.bokeh) || @testset "bokeh" begin
            @test plot_hpd(x_data, y_data_rep; backend = :bokeh) isa ArviZ.BokehPlot
        end
    end

    @testset "plot_elpd" begin
        plot_elpd(Dict("a" => data, "b" => data2))
        close(gcf())
        plot_elpd(Dict("a" => loo(data; pointwise=true), "b" => loo(data2; pointwise=true)))
        close(gcf())
        ispynull(ArviZ.bokeh) || @testset "bokeh" begin
            @test plot_elpd(Dict("a" => data, "b" => data2); backend = :bokeh) isa
                  ArviZ.BokehPlot
        end
    end

    @testset "plot_khat" begin
        l = loo(data; pointwise = true)
        plot_khat(l)
        close(gcf())
        ispynull(ArviZ.bokeh) || @testset "bokeh" begin
            @test plot_khat(l; backend = :bokeh) isa ArviZ.BokehPlot
        end
    end

    @testset "plot_loo_pit" begin
        plot_loo_pit(data; y = "obs")
        close(gcf())
        ispynull(ArviZ.bokeh) || @testset "bokeh" begin
            @test plot_loo_pit(data; y = "obs", backend = :bokeh) isa ArviZ.BokehPlot
        end
    end

    @testset "plot_loo_pit" begin
        plot_ppc(data)
        close(gcf())
        ispynull(ArviZ.bokeh) || @testset "bokeh" begin
            @test plot_ppc(data; backend = :bokeh) isa ArviZ.BokehPlot
        end
    end
end
