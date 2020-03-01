const has_bokeh_png_deps = false

function initialize_bokeh()
    ispynull(bokeh) || return
    try
        copy!(bokeh, pyimport_conda("bokeh", "bokeh", "conda-forge"))
    catch err
        copy!(bokeh, PyNULL())
        throw(err)
    end
end

# install dependencies for saving PNGs if using conda
function initialize_bokeh_png_deps()
    has_bokeh_png_deps && return
    try
        pyimport_conda("selenium", "selenium", "conda-forge")
        has_bokeh_png_deps = true
    catch err
        has_bokeh_png_deps = false
        throw(err)
    end
end

load_backend(::Val{:bokeh}) = initialize_bokeh()

convert_result(f, axis, ::Val{:bokeh}) = BokehPlot(axis)
function convert_result(f, axes::AbstractArray, ::Val{:bokeh})
    return BokehPlot(arviz.plots.backends.create_layout(axes))
end
function convert_result(::typeof(plot_joint), axes::AbstractArray, ::Val{:bokeh})
    return BokehPlot(arviz.plots.backends.create_layout(axes; force_layout = false))
end

"""
    BokehPlot(::PyObject)

Loose wrapper around a Bokeh figure, mostly used for dispatch.

In most cases, use one of the plotting functions with `backend=:bokeh` to create a
`BokehPlot` instead of using a constructor.
"""
struct BokehPlot
    o::PyObject
end

BokehPlot(plot::BokehPlot) = plot

@inline PyObject(plot::BokehPlot) = getfield(plot, :o)

Base.convert(::Type{BokehPlot}, o::PyObject) = BokehPlot(o)

Base.hash(plot::BokehPlot) = hash(PyObject(plot))

Base.propertynames(plot::BokehPlot) = propertynames(PyObject(plot))

function Base.getproperty(plot::BokehPlot, name::Symbol)
    o = PyObject(plot)
    name === :o && return o
    return getproperty(o, name)
end

function render_html(plot::BokehPlot, name = nothing)
    obj = PyObject(plot)
    return bokeh.embed.file_html(obj, bokeh.resources.CDN, name)
end

Base.display(::REPL.REPLDisplay, plot::BokehPlot) = bokeh.plotting.show(plot)

Base.show(io::IO, ::MIME"text/html", plot::BokehPlot) = print(io, render_html(plot))
function Base.show(
    io::IO,
    ::Union{MIME"application/juno+plotpane",MIME"application/prs.juno.plotpane+html"},
    plot::BokehPlot,
)
    return print(io, render_html(plot))
end
function Base.show(io::IO, ::MIME"juliavscode/html", plot::BokehPlot)
    return print(io, render_html(plot))
end
function Base.show(io::IO, ::MIME"image/png", plot::BokehPlot)
    initialize_bokeh_png_deps()
    fn = tempname() * ".png"
    bokeh.io.export_png(plot; filename = fn)
    png = read(fn, String)
    rm(fn)
    print(io, png)
end

"""
    write(io::IO, plot::BokehPlot)
    write(filename::AbstractString, plot::BokehPlot)

Write the HTML representation of the Bokeh plot to the I/O stream or file.
"""
Base.write(io::IO, plot::BokehPlot) = print(io, render_html(plot))
