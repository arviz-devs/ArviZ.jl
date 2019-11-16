"""
    use_style(style::String)
    use_style(style::Vector{String})

Use matplotlib style settings from a style specification `style`.

The style name of "default" is reserved for reverting back to the default style
settings.

ArviZ-specific styles are `["arviz-whitegrid", "arviz-darkgrid", "arviz-colors", "arviz-white"]`.
To see all available style specifications, use [`styles()`](@ref).

If a `Vector` of styles is provided, they are applied from first to last.
"""
use_style(style) = arviz.style.use(style)

"""
    styles() -> Vector{String}

Get all available matplotlib styles for use with [`use_style`](@ref)
"""
styles() = arviz.style.available

"""
    with_rc_context(f; rc = nothing, fname = nothing)

Execute the thunk `f` within a context controlled by rc params. To see
supported params, execute [`rc_params()`](@ref).

This allows one to do:

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

The `rc` dictionary takes precedence over the settings loaded from `fname`.
Passing a dictionary only is also valid.
"""
function with_rc_context(f; kwargs...)
    @pywith arviz.rc_context(; kwargs...) as _ begin
        f()
    end
end

"""
    rc_params() > Dict{String,Any}

Get the list of customizable `rc` params using [`with_rc_context`](@ref).
"""
rc_params() = Dict(k => v for (k, v) in ArviZ.arviz.rcParams)

"""
    with_interactive_backend(f; backend::Symbol = nothing)

Execute the thunk `f` in a temporary interactive context with the chosen
`backend`, or provide no arguments to use a default.

# Example

```julia
idata = load_arviz_data("centered_eight")
plot_posterior(idata) # inline
with_interactive_backend() do
    plot_density(idata) # interactive
end
plot_trace(idata) # inline
```
"""
function with_interactive_backend(f; backend = nothing)
    oldisint = PyPlot.isinteractive()
    oldgui = pygui()
    backend === nothing || pygui(Symbol(backend))
    pygui(true)
    f()
    pygui(oldisint)
    pygui(oldgui)
end

forwarddoc(f::Symbol) =
    "See documentation for [`arviz.$(f)`](https://arviz-devs.github.io/arviz/generated/arviz.$(f).html)."

forwardgetdoc(f::Symbol) = Docs.getdoc(getproperty(arviz, f))

"""
    @forwardfun f
    @forwardfun(f)

Wrap a function `arviz.f` in `f`, forwarding its docstrings.
"""
macro forwardfun(f)
    fdoc = forwarddoc(f)
    quote
        @doc $fdoc
        $(f)(args...; kwargs...) = arviz.$(f)(args...; kwargs...)

        Docs.getdoc(::typeof($(f))) = forwardgetdoc(Symbol($(f)))
    end |> esc
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

function popsubdict!(dict, names)
    (dict === nothing || names === nothing) && return nothing
    subdict = empty(dict)
    for k in names
        subdict[k] = pop!(dict, k)
    end
    return subdict
end

popsubdict!(dict, key::String) = popsubdict!(dict, [key])

snakecase(s) = replace(lowercase(s), " " => "_")

enforce_stat_types(dict) =
    Dict(k => get(sample_stats_types, k, eltype(v)).(v) for (k, v) in dict)
enforce_stat_types(::Nothing) = nothing
