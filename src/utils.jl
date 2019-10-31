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
