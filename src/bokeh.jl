function initialize_bokeh()
    ispynull(bokeh) || return

    try
        copy!(bokeh, pyimport_conda("bokeh", "bokeh", "conda-forge"))
        pytype_mapping(pyimport("bokeh.model").Model, BokehPlot)
        pytype_mapping(pyimport("bokeh.document").Document, BokehPlot)
    catch
    end
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

# We don't need to implement this `save` since FileIO defaults to calling the above
# `show` method.

"""
    write(io::IO, plot::BokehPlot)
    write(filename::AbstractString, plot::BokehPlot)

Write the HTML representation of the Bokeh plot to the I/O stream or file.
"""
Base.write(io::IO, plot::BokehPlot) = print(io, render_html(plot))
