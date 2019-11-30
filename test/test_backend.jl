using PyCall

@testset "matplotlib backend" begin
    idata = load_arviz_data("centered_eight")

    @test ArviZ.rc_params()["plot.backend"] == "matplotlib"
    @test plot_trace(idata) isa Array{PyObject}
end

if !ispynull(ArviZ.bokeh)
    @testset "bokeh backend" begin
        idata = load_arviz_data("centered_eight")

        with_rc_context(rc = Dict("plot.backend" => "bokeh")) do
            @test ArviZ.rc_params()["plot.backend"] == "bokeh"

            @test plot_trace(idata) isa ArviZ.BokehPlot
            @test plot_trace(idata; backend = "matplotlib") isa Array{PyObject}

            @testset "ArviZ.BokehPlot" begin
                plot = plot_trace(idata)

                pyobj = PyObject(plot)
                plot2 = convert(ArviZ.BokehPlot, pyobj)
                @test plot2 isa ArviZ.BokehPlot
                @test hash(plot) == hash(plot2)
                @test pyobj === PyObject(plot2)

                @test propertynames(plot) == propertynames(PyObject(plot))
                getproperty(plot, :__class__)

                @testset "MIME::\"$(mime)\"" for mime in ["text/html",]
                    text = repr(MIME(mime), plot)
                    @test text isa String
                    @test occursin("<body", text)
                end
            end
        end
    end
end
