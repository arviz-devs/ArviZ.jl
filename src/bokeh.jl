const has_bokeh_png_deps = false

function initialize_bokeh()
    Base.depwarn(
        "backend=\"bokeh\" is deprecated; use backend=\"matplotlib\" instead.", :backend
    )
    ispynull(bokeh) || return nothing
    try
        copy!(bokeh, _import_dependency("bokeh", "bokeh"; channel="conda-forge"))
    catch err
        copy!(bokeh, PyNULL())
        throw(err)
    end
    return nothing
end

# install dependencies for saving PNGs if using conda
function initialize_bokeh_png_deps()
    has_bokeh_png_deps && return nothing
    try
        _import_dependency("selenium", "selenium"; channel="conda-forge")
        has_bokeh_png_deps = true
    catch err
        has_bokeh_png_deps = false
        throw(err)
    end
    return nothing
end

load_backend(::Val{:bokeh}) = initialize_bokeh()

convert_result(f, axis, ::Val{:bokeh}) = BokehPlot(axis)
function convert_result(f, axes::AbstractArray, ::Val{:bokeh})
    return BokehPlot(arviz.plots.backends.create_layout(axes))
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

function render_html(plot::BokehPlot, name=nothing)
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
    image = bokeh.io.export.get_screenshot_as_png(plot)
    print(io, image._repr_png_())
    return nothing
end

"""
    write(io::IO, plot::BokehPlot)
    write(filename::AbstractString, plot::BokehPlot)

Write the HTML representation of the Bokeh plot to the I/O stream or file.
"""
Base.write(io::IO, plot::BokehPlot) = print(io, render_html(plot))
