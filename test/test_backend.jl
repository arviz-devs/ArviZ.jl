using PyCall

@testset "matplotlib backend" begin
    idata = load_arviz_data("centered_eight")

    backend = get(ArviZ.rcParams, "plot.backend", nothing)
    @test backend in ["matplotlib", nothing]
    @test plot_trace(idata) isa Array{PyObject}
    @test plot_trace(idata; backend = nothing) isa Array{PyObject}
end

if !ispynull(ArviZ.bokeh) && "plot.backend" in keys(ArviZ.rcParams)
    test_bokeh_png = try
        ArviZ.initialize_selenium()
        true
    catch
        @info "selenium not found. skipping tests for bokeh png"
        false
    end

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
                @test ArviZ.BokehPlot(plot) === plot
                @test hash(plot) == hash(plot2)
                @test pyobj === PyObject(plot2)

                @test propertynames(plot) == propertynames(PyObject(plot))
                @test occursin("bokeh", "$(getproperty(plot, :__class__))")

                test_bokeh_png && @testset "show MIME\"image/png\"" begin
                    io = IOBuffer(maxsize = 8)
                    show(io, MIME"image/png"(), plot)
                    @test take!(io) == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
                    close(io)
                end

                @testset "show MIME\"$(mime)\"" for mime in [
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
