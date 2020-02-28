"""
    RcParams <: AbstractDict

Class to contain ArviZ default parameters, with validation when setting items.

# Examples

```julia
julia> rcParams["plot.backend"]
"matplotlib"

julia> rcParams["plot.backend"] = "bokeh"
"bokeh"

julia> rcParams["plot.backend"]
"bokeh"
```
"""
struct RcParams{K,V} <: AbstractDict{K,V}
    o::PyObject
end
RcParams(obj) = RcParams{Any,Any}(obj)

@inline PyObject(r::RcParams) = getfield(r, :o)

Base.convert(::Type{RcParams{K,V}}, obj::PyObject) where {K,V} = RcParams{K,V}(obj)
Base.convert(::Type{RcParams}, obj::PyObject) = RcParams(obj)

const rcParams = RcParams{String,Any}(_rcParams)

@inline Base.length(r::RcParams) = py"len"(PyObject(r))
function Base.get(r::RcParams, k, default)
    haskey(r, k) && return PyObject(r).__getitem__(k)
    return default
end
function Base.setindex!(r::RcParams, v, k)
    try
        PyObject(r).__setitem__(k, v)
    catch e
        if e isa PyCall.PyError
            err = e.val
            if pyisinstance(err, py"ValueError")
                throw(ErrorException(err.args[1]))
            elseif pyisinstance(err, py"KeyError")
                throw(KeyError("$(k) is not a valid rc parameter (see keys(rcParams) for a list of valid parameters)"))
            end
        end
        throw(e)
    end
    return r
end

@inline Base.haskey(r::RcParams, k) = PyObject(r).__contains__(k)

function Base.iterate(r::RcParams, it)
    try
        pair = Pair(py"next"(it)...)
        return (pair, it)
    catch
        return nothing
    end
end
function Base.iterate(r::RcParams)
    items = PyObject(r).items()
    it = py"iter"(items)
    return Base.iterate(r, it)
end

"""
    with_rc_context(f; rc = nothing, fname = nothing)

Execute the thunk `f` within a context controlled by temporary rc params.

See [`rcParams`](@ref) for supported params or to modify params long-term.

# Examples

```julia
with_rc_context(fname = "pystan.rc") do
    idata = load_arviz_data("radon")
    plot_posterior(idata; var_names=["gamma"])
end
```

The plot would have settings from `pystan.rc`.

A dictionary can also be passed to the context manager:

```julia
with_rc_context(rc = Dict("plot.max_subplots" => 1), fname = "pystan.rc") do
    idata = load_arviz_data("radon")
    plot_posterior(idata, var_names=["gamma"])
end
```

The `rc` dictionary takes precedence over the settings loaded from `fname`. Passing a
dictionary only is also valid.
"""
function with_rc_context(f; kwargs...)
    @pywith arviz.rc_context(; kwargs...) as _ begin
        return f()
    end
end
