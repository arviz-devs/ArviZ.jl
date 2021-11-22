using PyCall

@testset "matplotlib backend" begin
    idata = load_arviz_data("centered_eight")

    backend = get(ArviZ.rcParams, "plot.backend", nothing)
    @test backend in ["matplotlib", nothing]
    @test plot_trace(idata) isa Array{PyObject}
    @test plot_trace(idata; backend=nothing) isa Array{PyObject}
end

if !ispynull(ArviZ.bokeh) && "plot.backend" in keys(ArviZ.rcParams)
    try
        ArviZ.initialize_bokeh_png_deps()
    catch
        @info "bokeh png depencies not found. Skipping tests for bokeh png"
    end

    @testset "bokeh backend" begin
        idata = load_arviz_data("centered_eight")

        with_rc_context(; rc=Dict("plot.backend" => "bokeh")) do
            backend = get(ArviZ.rcParams, "plot.backend", nothing)
            @test backend == "bokeh"

            @test plot_trace(idata) isa ArviZ.BokehPlot
            @test plot_trace(idata; backend="matplotlib") isa Array{PyObject}

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

                ArviZ.has_bokeh_png_deps && @testset "show MIME\"image/png\"" begin
                    fn = tempname() * ".png"
                    open(fn, "w") do s
                        return show(s, MIME"image/png"(), plot)
                    end
                    bytes = read(fn)
                    @test bytes[1:8] ==
                        UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
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
            return nothing
        end
    end
end
