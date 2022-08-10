@forwardfun extract_dataset
convert_result(::typeof(extract_dataset), result, args...) = convert(Dataset, result)

function convert_to_inference_data(filename::AbstractString; kwargs...)
    return from_netcdf(filename)
end

"""
    default_var_name(data) -> Symbol

Return the default name for the variable whose values are stored in `data`.
"""
default_var_name(data) = :x
function default_var_name(data::DimensionalData.AbstractDimArray)
    name = DimensionalData.name(data)
    name isa Symbol && return name
    name isa AbstractString && !isempty(name) && return Symbol(name)
    return default_var_name(parent(data))
end

@forwardfun load_arviz_data

@forwardfun to_netcdf
@forwardfun from_netcdf
@forwardfun from_json
@forwardfun from_dict
@forwardfun from_cmdstan
@forwardfun from_cmdstanpy
@forwardfun from_emcee
@forwardfun from_pymc3
@forwardfun from_pyro
@forwardfun from_numpyro
@forwardfun from_pystan

@doc forwarddoc(:concat) concat

function concat(data::InferenceData...; kwargs...)
    return arviz.concat(data...; inplace=false, kwargs...)
end

Docs.getdoc(::typeof(concat)) = forwardgetdoc(:concat)

"""
    merge(data::InferenceData, others::InferenceData...) -> InferenceData

Merge [`InferenceData`](@ref) objects.

The result contains all groups in `data` and `others`.
If a group appears more than once, the one that occurs first is kept.

See [`concat`](@ref)
"""
function Base.merge(data::InferenceData, others::InferenceData...)
    return InferenceData(Base.merge(groups(data), map(groups, others)...))
end

function rekey(data::InferenceData, keymap)
    groups_old = groups(data)
    names_new = map(k -> get(keymap, k, k), propertynames(groups_old))
    groups_new = NamedTuple{names_new}(Tuple(groups_old))
    return InferenceData(groups_new)
end
