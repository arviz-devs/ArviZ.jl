struct BokehPlot
    o
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

function Base.show(io::IO, ::MIME"application/prs.juno.plotpane+html", plot::BokehPlot)
    return print(io, render_html(plot))
end

function Base.show(io::IO, ::MIME"juliavscode/html", plot::BokehPlot)
    return print(io, render_html(plot))
end
