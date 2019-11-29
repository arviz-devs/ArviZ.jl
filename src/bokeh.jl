struct BokehPlot
    o
end

@inline PyObject(plot::BokehPlot) = getfield(plot, :o)

Base.convert(::Type{BokehPlot}, o::PyObject) = BokehPlot(o)

Base.hash(plot::BokehPlot) = hash(PyObject(data))

Base.propertynames(plot::BokehPlot) = propertynames(PyObject(data))

function Base.getproperty(plot::BokehPlot, name::Symbol)
    o = PyObject(plot)
    name === :o && return o
    return getproperty(o, name)
end

