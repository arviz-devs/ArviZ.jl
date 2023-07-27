const sample_stats_eltypes = (
    lp=Float64,
    step_size=Float64,
    step_size_nom=Float64,
    tree_depth=Int,
    n_steps=Int,
    diverging=Bool,
    energy=Float64,
    energy_error=Float64,
    max_energy_error=Float64,
    int_time=Float64,
)

# Replace `missing` values with `NaN` and do type inference on the result
replacemissing(x) = map(identity, replace(x, missing => NaN))
replacemissing(x::AbstractArray{<:AbstractArray}) = map(replacemissing, x)
@inline replacemissing(x::AbstractArray{<:Real}) = x
@inline replacemissing(::Missing) = NaN
@inline replacemissing(x::Number) = x

enforce_stat_eltypes(stats) = convert_to_eltypes(stats, sample_stats_eltypes)

function convert_to_eltypes(data::Dict, data_eltypes)
    return Dict(k => convert(Array{get(data_eltypes, k, eltype(v))}, v) for (k, v) in data)
end
function convert_to_eltypes(data::NamedTuple, data_eltypes)
    return NamedTuple(
        k => convert(Array{get(data_eltypes, k, eltype(v))}, v) for (k, v) in pairs(data)
    )
end
