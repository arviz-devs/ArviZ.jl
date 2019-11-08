"""
    @forwardfun f
    @forwardfun(f)

Wrap a function `arviz.f` in `f`
"""
macro forwardfun(f)
    quote
        @__doc__ function $(f)(args...; kwargs...)
            arviz.$(f)(args...; kwargs...)
        end

        Base.Docs.getdoc(::typeof($(f))) = Base.Docs.getdoc(arviz.$(f))
    end |> esc
end

"""
    styles()

Get all available matplotlib styles.
"""
styles() = arviz.style.available

"""
    use_style(style::Union{String,Vector{String}})

Use matplotlib style settings from a style specification `style`.

The style name of "default" is reserved for reverting back to the default style
settings.

ArviZ-specific styles are "arviz-whitegrid", "arviz-darkgrid", "arviz-colors",
and "arviz-white". To see all available style specifications, use
[`styles`](@ref).

If a `Vector` of styles is provided, they are applied from first to last.
"""
use_style(style) = arviz.style.use(style)

"""
    rc_context(f, rc = nothing, fname = nothing)

Execute the thunk `f` within a context controlled by rc params. To see
supported params, execute `rc_params()`.

This allows one to do:

```julia
rc_context(fname = "pystan.rc") do
    idata = load_arviz_data("radon")
    plot_posterior(idata; var_names=["gamma"])
end
```

The plot would have settings from `pystan.rc`.

A dictionary can also be passed to the context manager:

```julia
rc_context(rc = Dict("plot.max_subplots" => 1), fname = "pystan.rc") do
    idata = load_arviz_data("radon")
    plot_posterior(idata, var_names=["gamma"])
end
```

The `rc` dictionary takes precedence over the settings loaded from `fname`.
Passing a dictionary only is also valid.
"""
function rc_context(f, args...; kwargs...)
    @pywith arviz.rc_context(args...; kwargs...) as _ begin
        f()
    end
end

rc_params() = arviz.rcParams()

"""
    interactive_backend(f, backend::Union{Symbol,Nothing} = nothing)

Execute the thunk `f` in a temporary interactive context of choice, or provide
no arguments to use a default.

# Example

```julia
idata = load_arviz_data("centered_eight")
plot_posterior(idata) # inline
interactive_backend() do
    plot_density(idata) # interactive
end
plot_trace(idata) # inline
```
"""
function interactive_backend(f, backend = nothing)
    oldisint = PyPlot.isinteractive()
    oldgui = pygui()
    backend === nothing || pygui(Symbol(backend))
    pygui(true)
    f()
    pygui(oldisint)
    pygui(oldgui)
end

"""
    replacemissing(x)

Replace `missing` values with `NaN` and do type inference on the result.
"""
replacemissing(x) = map(identity, replace(x, missing => NaN))
replacemissing(x::AbstractArray{<:Real}) = x
replacemissing(x::Missing) = NaN

function rekey(d, keymap)
    dnew = empty(d)
    for (k, v) in d
        knew = get(keymap, k, k)
        haskey(dnew, knew) && throw(ArgumentError("$knew in `keymap` is already in `d`."))
        dnew[knew] = d[k]
    end
    return dnew
end

removekeys!(dict, keys) = map(k -> delete!(dict, k), keys)

function subdict(dict, keys)
    d = empty(dict)
    for k in unique(keys)
        if k in keys(dict)
            d[k] = dict[v]
        end
    end
    return d
end

function popsubdict!(dict, names)
    (isnothing(dict) || isnothing(names)) && return nothing
    subdict = empty(dict)
    for k in names
        subdict[k] = pop!(dict, k)
    end
    return subdict
end

popsubdict!(dict, key::String) = popsubdict!(dict, [key])

snakecase(s) = replace(lowercase(s), " " => "_")

function indexify(params::Vector{String})
    d = Dict{String,String}()
    for name in params
        indexed_name = replace(name, r"\.(\w+)" => s"[\1]")
        indexed_name == name && continue
        d[name] = indexed_name
    end
    return d
end
