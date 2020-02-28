using PyCall

@testset "matplotlib backend" begin
    idata = load_arviz_data("centered_eight")

    backend = get(ArviZ.rcParams, "plot.backend", nothing)
    @test backend in ["matplotlib", nothing]
    @test plot_trace(idata) isa Array{PyObject}
end

if !ispynull(ArviZ.bokeh) && "plot.backend" in keys(ArviZ.rcParams)
    @testset "bokeh backend" begin
        idata = load_arviz_data("centered_eight")

        with_rc_context(rc = Dict("plot.backend" => "bokeh")) do
            backend = get(ArviZ.rcParams, "plot.backend", nothing)
            @test backend == "bokeh"

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
                @test occursin("bokeh", "$(getproperty(plot, :__class__))")

                @testset "show MIME::\"$(mime)\"" for mime in [
                    "text/html",
                    "juliavscode/html",
                    "application/juno+plotpane",
                    "application/prs.juno.plotpane+html",
                ]
                    text = sprint(show, MIME(mime), plot)
                    @test text isa String
                    @test occursin("<body", text)
                end

                @testset "write html" begin
                    text = sprint(write, plot)
                    @test text isa String
                    @test occursin("<body", text)
                end
            end
        end
    end
end
