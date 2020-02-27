"""
    use_style(style::String)
    use_style(style::Vector{String})

Use matplotlib style settings from a style specification `style`.

The style name of "default" is reserved for reverting back to the default style settings.

ArviZ-specific styles are
`["arviz-whitegrid", "arviz-darkgrid", "arviz-colors", "arviz-white"]`.
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

Execute the thunk `f` within a context controlled by rc params. To see supported params,
execute [`rc_params()`](@ref).

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

The `rc` dictionary takes precedence over the settings loaded from `fname`. Passing a
dictionary only is also valid.
"""
function with_rc_context(f; kwargs...)
    @pywith arviz.rc_context(; kwargs...) as _ begin
        return f()
    end
end

"""
    rc_params() -> Dict{String,Any}

Get the list of customizable `rc` params using [`with_rc_context`](@ref).
"""
rc_params() = Dict(k => v for (k, v) in ArviZ.arviz.rcParams)

"""
    with_interactive_backend(f; backend::Symbol = nothing)

Execute the thunk `f` in a temporary interactive context with the chosen `backend`, or
provide no arguments to use a default.

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
    ret = f()
    pygui(oldisint)
    pygui(oldgui)
    return ret
end

"""
    convert_arguments(f, args...; kwargs...) -> NTuple{2}

Convert arguments to the function `f` before calling.

This function is used primarily for pre-processing arguments within macros before sending
to arviz.
"""
convert_arguments(::Any, args...; kwargs...) = args, kwargs

"""
    convert_result(f, result)

Convert result of the function `f` before returning.

This function is used primarily for post-processing outputs of arviz before returning.
"""
convert_result(::Any, result) = result

forwarddoc(f::Symbol) =
    "See documentation for [`arviz.$(f)`](https://arviz-devs.github.io/arviz/generated/arviz.$(f).html)."

forwardgetdoc(f::Symbol) = Docs.getdoc(getproperty(arviz, f))

"""
    @forwardfun f
    @forwardfun(f)

Wrap a function `arviz.f` in `f`, forwarding its docstrings.

Use [`convert_arguments`](@ref) and [`convert_result`](@ref) to customize what is passed to
and returned from `f`.
"""
macro forwardfun(f)
    fdoc = forwarddoc(f)
    quote
        @doc $fdoc
        function $(f)(args...; kwargs...)
            args, kwargs = convert_arguments($(f), args...; kwargs...)
            result = arviz.$(f)(args...; kwargs...)
            return convert_result($(f), result)
        end

        Docs.getdoc(::typeof($(f))) = forwardgetdoc(Symbol($(f)))
    end |> esc
end

"""
    @forwardplotfun f
    @forwardplotfun(f)

Wrap a plotting function `arviz.f` in `f`, forwarding its docstrings.

This macro also ensures that outputs for the different backends are correctly handled.
Use [`convert_arguments`](@ref) and [`convert_result`](@ref) to customize what is passed to
and returned from `f`.
"""
macro forwardplotfun(f)
    fdoc = forwarddoc(f)
    quote
        @doc $fdoc
        function $(f)(
            args...;
            backend = get(rc_params(), "plot.backend", nothing),
            kwargs...,
        )
            backend === nothing && return arviz.$(f)(args...; kwargs...)
            return $(f)(Val(Symbol(backend)), args...; kwargs...)
        end

        function $(f)(::Val, args...; kwargs...)
            args, kwargs = convert_arguments($(f), args...; kwargs...)
            result = arviz.$(f)(args...; kwargs...)
            return convert_result($(f), result)
        end

        function $(f)(::Val{:matplotlib}, args...; kwargs...)
            args, kwargs = convert_arguments($(f), args...; kwargs...)
            kwargs = merge(kwargs, Dict(:backend => "matplotlib"))
            try
                result = arviz.$(f)(args...; kwargs...)
                return convert_result($(f), result)
            catch e
                e isa PyCall.PyError || rethrow(e)
                pop!(kwargs, :backend)
                result = arviz.$(f)(args...; kwargs...)
                return convert_result($(f), result)
            end
        end

        function $(f)(::Val{:bokeh}, args...; kwargs...)
            args, kwargs = convert_arguments($(f), args...; kwargs...)
            kwargs = merge(kwargs, Dict(:backend => "bokeh", :show => false))
            plots = arviz.$(f)(args...; kwargs...)
            plots isa BokehPlot && return plots
            return bokeh.plotting.gridplot(plots)
        end

        Docs.getdoc(::typeof($(f))) = forwardgetdoc(Symbol($(f)))
    end |> esc
end

# Replace `missing` values with `NaN` and do type inference on the result
replacemissing(x) = map(identity, replace(x, missing => NaN))
@inline replacemissing(x::AbstractArray{<:Real}) = x
@inline replacemissing(x::Missing) = NaN
@inline replacemissing(x::Number) = x

# Convert python types to Julia types if possible
@inline frompytype(x) = x
@inline frompytype(x::PyObject) = PyAny(x)
frompytype(x::AbstractArray{PyObject}) = map(frompytype, x)
frompytype(x::AbstractArray{Any}) = map(frompytype, x)
frompytype(x::AbstractArray{<:AbstractArray}) = map(frompytype, x)

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

@inline _asarray(x) = [x]
@inline _asarray(x::AbstractArray) = x

enforce_stat_types(dict) =
    Dict(k => get(sample_stats_types, k, eltype(v)).(v) for (k, v) in dict)
enforce_stat_types(::Nothing) = nothing

"""
    todataframes(df; index_name = nothing) -> DataFrames.DataFrame

Convert a Python `Pandas.DataFrame` or `Pandas.Series` into a `DataFrames.DataFrame`.

If `index_name` is not `nothing`, the index is converted into a column with `index_name`.
Otherwise, it is discarded.
"""
function todataframes(df::Pandas.DataFrame; index_name = nothing)
    col_vals = map(columns(df)) do name
        series = getindex(df, name)
        vals = PyObject(series).values
        return Symbol(name) => frompytype(vals)
    end
    if index_name !== nothing
        index_vals = frompytype(Array(Pandas.index(df)))
        col_vals = [Symbol(index_name) => index_vals; col_vals]
    end
    return DataFrames.DataFrame(col_vals)
end
function todataframes(s::Pandas.Series; kwargs...)
    colnames = map(Symbol, Pandas.index(s))
    colvals = map(x -> [frompytype(x)], PyObject(s).values)
    return DataFrames.DataFrame(colvals, colnames)
end
function todataframes(df::PyObject; kwargs...)
    if pyisinstance(df, Pandas.pandas_raw.Series)
        return todataframes(Pandas.Series(df); kwargs...)
    end
    return todataframes(Pandas.DataFrame(df); kwargs...)
end

"""
    topandas(::Type{Pandas.DataFrame}, df; index_name = nothing) -> Pandas.DataFrame
    topandas(::Type{Pandas.Series}, df) -> Pandas.Series
    topandas(::Val{:ELPDData}, df) -> Pandas.Series

Convert a `DataFrames.DataFrame` to the specified Pandas type.

If `index_name` is not `nothing`, the corresponding column is made the index of the
returned dataframe.
"""
function topandas(::Type{Pandas.DataFrame}, df; index_name = nothing)
    df = DataFrames.DataFrame(df)
    colnames = names(df)
    rowvals = map(Array, eachrow(df))
    pdf = Pandas.DataFrame(rowvals; columns = colnames)
    index_name !== nothing && set_index(pdf, index_name; inplace = true)
    return pdf
end
function topandas(::Type{Pandas.Series}, df)
    df = DataFrames.DataFrame(df)
    rownames = names(df)
    colvals = Array(first(eachrow(df)))
    return Pandas.Series(colvals, rownames)
end
function topandas(::Val{:ELPDData}, df)
    df = DataFrames.DataFrame(df)
    rownames = names(df)
    colvals = Array(first(eachrow(df)))
    elpd_data = ArviZ.arviz.stats.ELPDData(colvals, rownames)
    return Pandas.Series(elpd_data)
end
