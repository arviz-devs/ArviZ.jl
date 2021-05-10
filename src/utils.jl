@doc doc"""
    with_interactive_backend(f; backend::Symbol = nothing)

Execute the thunk `f` in a temporary interactive context with the chosen `backend`, or
provide no arguments to use a default.

# Examples

```julia
idata = load_arviz_data("centered_eight")
plot_posterior(idata) # inline
with_interactive_backend() do
    plot_density(idata) # interactive
end
plot_trace(idata) # inline
```
"""
with_interactive_backend

function with_interactive_backend(f; backend=nothing)
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
    use_style(style::String)
    use_style(style::Vector{String})

Use matplotlib style settings from a style specification `style`.

The style name of "default" is reserved for reverting back to the default style settings.

ArviZ-specific styles are
`["arviz-whitegrid", "arviz-darkgrid", "arviz-colors", "arviz-white"]`.
To see all available style specifications, use [`styles()`](@ref).

If a `Vector` of styles is provided, they are applied from first to last.
"""
use_style(style) = plt.style.use(style)

"""
    styles() -> Vector{String}

Get all available matplotlib styles for use with [`use_style`](@ref)
"""
styles() = plt.style.available

"""
    convert_arguments(f, args...; kwargs...) -> NTuple{2}

Convert arguments to the function `f` before calling.

This function is used primarily for pre-processing arguments within macros before sending
to arviz.
"""
convert_arguments(::Any, args...; kwargs...) = args, kwargs

"""
    convert_result(f, result, args...)

Convert result of the function `f` before returning.

This function is used primarily for post-processing outputs of arviz before returning.
The `args` are primarily used for dispatch.
"""
convert_result(f, result, args...) = result

load_backend(backend) = nothing

function forwarddoc(f::Symbol)
    return "See documentation for [`arviz.$(f)`](https://arviz-devs.github.io/arviz/api/generated/arviz.$(f).html)."
end

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
    return esc(
        quote
            @doc $fdoc $f

            function $(f)(args...; kwargs...)
                args, kwargs = convert_arguments($(f), args...; kwargs...)
                result = arviz.$(f)(args...; kwargs...)
                return convert_result($(f), result)
            end

            Docs.getdoc(::typeof($(f))) = forwardgetdoc(Symbol($(f)))
        end,
    )
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
    return esc(
        quote
            @doc $fdoc $f

            function $(f)(args...; backend=nothing, kwargs...)
                if backend === nothing
                    backend = get(rcParams, "plot.backend", nothing)
                end
                backend = Symbol(backend)
                backend_val = Val(backend)
                load_backend(backend_val)
                args, kwargs = convert_arguments($(f), args...; kwargs...)
                result = arviz.$(f)(args...; kwargs..., backend=backend)
                return convert_result($(f), result, backend_val)
            end

            Docs.getdoc(::typeof($(f))) = forwardgetdoc(Symbol($(f)))
        end,
    )
end

# Replace `missing` values with `NaN` and do type inference on the result
replacemissing(x) = map(identity, replace(x, missing => NaN))
replacemissing(x::AbstractArray{<:AbstractArray}) = map(replacemissing, x)
@inline replacemissing(x::AbstractArray{<:Real}) = x
@inline replacemissing(x::Missing) = NaN
@inline replacemissing(x::Number) = x

"""
    flatten(x)

If `x` is an array of arrays, flatten into a single array whose dimensions are ordered with
dimensions of the outermost container first and innermost container last.
"""
flatten(x) = x
flatten(x::AbstractArray{T}) where {T<:Number} = convert(Array, x)
function flatten(x::AbstractArray{S}) where {T<:Number,N,S<:AbstractArray{T,N}}
    ret = Array{T}(undef, (size(x)..., size(x[1])...))
    for k in keys(x)
        setindex!(ret, x[k], k, (Colon() for _ in 1:N)...)
    end
    return ret
end

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

_asstringkeydict(x) = Dict(String(k) => v for (k, v) in pairs(x))
_asstringkeydict(x::Dict{String}) = x
_asstringkeydict(::Nothing) = Dict{String,Any}()

function enforce_stat_types(dict)
    return Dict(k => get(sample_stats_types, k, eltype(v)).(v) for (k, v) in dict)
end
enforce_stat_types(::Nothing) = nothing

"""
    todataframes(df; index_name = nothing) -> DataFrames.DataFrame

Convert a Python `pandas.DataFrame` or `pandas.Series` into a `DataFrames.DataFrame`.

If `index_name` is not `nothing`, the index is converted into a column with `index_name`.
Otherwise, it is discarded.
"""
function todataframes(::Val{:DataFrame}, df::PyObject; index_name=nothing)
    initialize_pandas()
    col_vals = map(df.columns) do name
        series = py"$(df)[$(name)]"
        vals = series.values
        return Symbol(name) => frompytype(vals)
    end
    if index_name !== nothing
        index_vals = frompytype(df.index.values)
        col_vals = [Symbol(index_name) => index_vals; col_vals]
    end
    return DataFrames.DataFrame(col_vals)
end
function todataframes(::Val{:Series}, series::PyObject; kwargs...)
    initialize_pandas()
    colnames = map(i -> Symbol(frompytype(i)), series.index)
    colvals = map(x -> [frompytype(x)], series.values)
    return DataFrames.DataFrame(colvals, colnames)
end
function todataframes(df::PyObject; kwargs...)
    initialize_pandas()
    if pyisinstance(df, pandas.Series)
        return todataframes(Val(:Series), df; kwargs...)
    end
    return todataframes(Val(:DataFrame), df; kwargs...)
end

"""
    topandas(::Type{:DataFrame}, df; index_name = nothing) -> PyObject
    topandas(::Type{:Series}, df) -> PyObject
    topandas(::Val{:ELPDData}, df) -> PyObject

Convert a `DataFrames.DataFrame` to the specified pandas type.

If `index_name` is not `nothing`, the corresponding column is made the index of the
returned dataframe.
"""
function topandas(::Val{:DataFrame}, df; index_name=nothing)
    initialize_pandas()
    df = DataFrames.DataFrame(df)
    colnames = names(df)
    rowvals = map(Array, eachrow(df))
    pdf = pandas.DataFrame(rowvals; columns=colnames)
    index_name !== nothing && pdf.set_index(index_name; inplace=true)
    return pdf
end
function topandas(::Val{:Series}, df)
    initialize_pandas()
    df = DataFrames.DataFrame(df)
    rownames = names(df)
    colvals = Array(first(eachrow(df)))
    return pandas.Series(colvals, rownames)
end
function topandas(::Val{:ELPDData}, df)
    initialize_pandas()
    df = DataFrames.DataFrame(df)
    rownames = names(df)
    colvals = Array(first(eachrow(df)))
    return ArviZ.arviz.stats.ELPDData(colvals, rownames)
end
