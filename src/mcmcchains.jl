const turing_key_map = Dict(
    "hamiltonian_energy" => "energy",
    "hamiltonian_energy_error" => "energy_error",
    "is_adapt" => "tune",
    "max_hamiltonian_energy_error" => "max_energy_error",
    "nom_step_size" => "step_size_nom",
    "numerical_error" => "diverging",
)
const stan_key_map = Dict(
    "accept_stat__" => "acceptance_rate",
    "divergent__" => "diverging",
    "energy__" => "energy",
    "lp__" => "lp",
    "n_leapfrog__" => "n_steps",
    "stepsize__" => "step_size",
    "treedepth__" => "tree_depth",
)
const stats_key_map = merge(turing_key_map, stan_key_map)

"""
    reshape_values(x::AbstractArray) -> AbstractArray

Convert from `MCMCChains` variable values with dimensions `(ndraw, size..., nchain)` to
ArviZ's expected `(nchain, ndraw, size...)`.
"""
reshape_values(x::AbstractArray{T,N}) where {T,N} = permutedims(x, (N, 1, 2:(N - 1)...))

headtail(x) = x[1], x[2:end]

function split_locname(name)
    name = replace(name, r"[\[,]" => '.')
    name = replace(name, ']' => "")
    name, loc = headtail(split(name, '.'))
    isempty(loc) && return name, ()
    loc = tryparse.(Int, loc)
    Nothing <: eltype(loc) && return name, ()
    return name, tuple(loc...)
end

function varnames_locs_dict(loc_names, loc_str_to_old)
    vars_to_locs = Dict()
    for loc_name in loc_names
        var_name, loc = split_locname(loc_name)
        if var_name ∉ keys(vars_to_locs)
            vars_to_locs[var_name] = ([loc_str_to_old[loc_name]], [loc])
        else
            push!(vars_to_locs[var_name][1], loc_str_to_old[loc_name])
            push!(vars_to_locs[var_name][2], loc)
        end
    end
    return vars_to_locs
end

function attributes_dict(chns::Chains)
    info = delete(chns.info, :hashedsummary)
    return Dict{String,Any}((string(k), v) for (k, v) in pairs(info))
end
attributes_dict(::Nothing) = Dict()

function section_dict(chns::Chains, section)
    ndraws, _, nchains = size(chns)
    loc_names_old = getfield(chns.name_map, section) # old may be Symbol or String
    loc_names = string.(loc_names_old)
    loc_str_to_old = Dict(
        name_str => name_old for (name_str, name_old) in zip(loc_names, loc_names_old)
    )
    vars_to_locs = varnames_locs_dict(loc_names, loc_str_to_old)
    vars_to_arrays = Dict{String,Array}()
    for (var_name, names_locs) in vars_to_locs
        loc_names, locs = names_locs
        max_loc = maximum(reduce(hcat, [loc...] for loc in locs); dims=2)
        ndim = length(max_loc)
        sizes = tuple(max_loc...)

        oldarr = reshape_values(replacemissing(convert(Array, chns.value[:, loc_names, :])))
        if iszero(ndim)
            arr = dropdims(oldarr; dims=3)
        else
            arr = Array{Union{typeof(NaN),eltype(oldarr)}}(undef, nchains, ndraws, sizes...)
            fill!(arr, NaN)
            for i in eachindex(locs)
                arr[:, :, locs[i]...] = oldarr[:, :, i]
            end
        end
        vars_to_arrays[var_name] = arr
    end
    return vars_to_arrays
end

function chains_to_dict(
    chns::Chains; ignore=String[], section=:parameters, rekey_fun=identity
)
    section in sections(chns) || return Dict()
    chns_dict = section_dict(chns, section)
    removekeys!(chns_dict, ignore)
    return rekey_fun(chns_dict)
end
chains_to_dict(::Nothing; kwargs...) = nothing

"""
    convert_to_inference_data(obj::Chains; group = :posterior, kwargs...) -> InferenceData

Convert the chains `obj` to an [`InferenceData`](@ref) with the specified `group`.

Remaining `kwargs` are forwarded to [`from_mcmcchains`](@ref).
"""
function convert_to_inference_data(chns::Chains; group=:posterior, kwargs...)
    group = Symbol(group)
    group === :posterior && return from_mcmcchains(chns; kwargs...)
    return from_mcmcchains(; group => chns)
end

@doc doc"""
    from_mcmcchains(posterior::Chains; kwargs...) -> InferenceData
    from_mcmcchains(; kwargs...) -> InferenceData
    from_mcmcchains(
        posterior::Chains,
        posterior_predictive::Any,
        predictions::Any,
        log_likelihood::Any;
        kwargs...
    ) -> InferenceData

Convert data in an `MCMCChains.Chains` format into an [`InferenceData`](@ref).

Any keyword argument below without an an explicitly annotated type above is allowed, so long
as it can be passed to [`convert_to_inference_data`](@ref).

# Arguments

- `posterior::Chains`: Draws from the posterior

# Keywords

- `posterior_predictive::Any=nothing`: Draws from the posterior predictive distribution or
    name(s) of predictive variables in `posterior`
- `predictions::Any=nothing`: Out-of-sample predictions for the posterior.
- `prior::Any=nothing`: Draws from the prior
- `prior_predictive::Any=nothing`: Draws from the prior predictive distribution or name(s)
    of predictive variables in `prior`
- `observed_data::Dict{String,Array}=nothing`: Observed data on which the `posterior` is
    conditional. It should only contain data which is modeled as a random variable. Keys are
    parameter names and values.
- `constant_data::Dict{String,Array}=nothing`: Model constants, data included in the model
    which is not modeled as a random variable. Keys are parameter names and values.
- `predictions_constant_data::Dict{String,Array}=nothing`: Constants relevant to the model
     predictions (i.e. new `x` values in a linear regression).
- `log_likelihood::Any=nothing`: Pointwise log-likelihood for the data. It is recommended
     to use this argument as a dictionary whose keys are observed variable names and whose
     values are log likelihood arrays.
- `log_likelihood::String=nothing`: Name of variable in `posterior` with log likelihoods
- `library=MCMCChains`: Name of library that generated the chains
- `coords::Dict{String,Vector}=nothing`: Map from named dimension to named indices
- `dims::Dict{String,Vector{String}}=nothing`: Map from variable name to names of its
    dimensions

# Returns

- `InferenceData`: The data with groups corresponding to the provided data
"""
from_mcmcchains

function from_mcmcchains(
    posterior,
    posterior_predictive,
    predictions,
    log_likelihood;
    library=MCMCChains,
    kwargs...,
)
    kwargs = convert(Dict, merge((; dims=nothing), kwargs))
    library = string(library)
    rekey_fun = d -> rekey(d, stats_key_map)

    # Convert chains to dicts
    post_dict = chains_to_dict(posterior)
    stats_dict = chains_to_dict(posterior; section=:internals, rekey_fun=rekey_fun)
    stats_dict = enforce_stat_types(stats_dict)

    all_idata = InferenceData()
    for (group, group_data) in [
        :posterior_predictive => posterior_predictive,
        :predictions => predictions,
        :log_likelihood => log_likelihood,
    ]
        group_data === nothing && continue
        if group_data isa Union{Symbol,String}
            group_data = [string(group_data)]
        end
        if group_data isa Union{AbstractVector{Symbol},NTuple{N,Symbol} where {N}}
            group_data = map(string, group_data)
        end
        if group_data isa Union{AbstractVector{String},NTuple{N,String} where {N}}
            group_data = popsubdict!(post_dict, group_data)
        end
        group_dataset = convert_to_dataset(group_data; library=library, kwargs...)
        setattribute!(group_dataset, "inference_library", library)
        concat!(all_idata, InferenceData(; group => group_dataset))
    end

    attrs = attributes_dict(posterior)
    attrs = merge(attrs, Dict("inference_library" => library))
    kwargs = convert(Dict, merge((; attrs=attrs, dims=nothing), kwargs))
    post_idata = _from_dict(post_dict; sample_stats=stats_dict, kwargs...)
    concat!(all_idata, post_idata)
    return all_idata
end
function from_mcmcchains(
    posterior=nothing;
    posterior_predictive=nothing,
    predictions=nothing,
    prior=nothing,
    prior_predictive=nothing,
    observed_data=nothing,
    constant_data=nothing,
    predictions_constant_data=nothing,
    log_likelihood=nothing,
    library=MCMCChains,
    kwargs...,
)
    kwargs = convert(Dict, merge((; dims=nothing, coords=nothing), kwargs))

    all_idata = from_mcmcchains(
        posterior,
        posterior_predictive,
        predictions,
        log_likelihood;
        library=library,
        kwargs...,
    )

    if prior !== nothing
        pre_prior_idata = convert_to_inference_data(
            prior; posterior_predictive=prior_predictive, library=library, kwargs...
        )
        prior_idata = rekey(
            pre_prior_idata,
            Dict(
                :posterior => :prior,
                :posterior_predictive => :prior_predictive,
                :sample_stats => :sample_stats_prior,
            ),
        )
        concat!(all_idata, prior_idata)
    end

    for (group, group_data) in [
        :observed_data => observed_data,
        :constant_data => constant_data,
        :predictions_constant_data => predictions_constant_data,
    ]
        group_data === nothing && continue
        group_dataset = convert_to_constant_dataset(group_data; library=library, kwargs...)
        concat!(all_idata, InferenceData(; group => group_dataset))
    end

    return all_idata
end

"""
    from_cmdstan(posterior::Chains; kwargs...) -> InferenceData

Call [`from_mcmcchains`](@ref) on output of `CmdStan`.
"""
function from_cmdstan(posterior::Chains; kwargs...)
    return from_mcmcchains(posterior; library="CmdStan", kwargs...)
end
