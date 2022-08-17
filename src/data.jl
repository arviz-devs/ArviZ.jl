@forwardfun extract_dataset
convert_result(::typeof(extract_dataset), result, args...) = convert(Dataset, result)

function convert_to_inference_data(filename::AbstractString; kwargs...)
    return from_netcdf(filename)
end

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
