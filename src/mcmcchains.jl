const turing_key_map = Dict(
    :hamiltonian_energy => :energy,
    :hamiltonian_energy_error => :energy_error,
    :is_adapt => :tune,
    :max_hamiltonian_energy_error => :max_energy_error,
    :nom_step_size => :step_size_nom,
    :numerical_error => :diverging,
)
const stan_key_map = Dict(
    :accept_stat__ => :acceptance_rate,
    :divergent__ => :diverging,
    :energy__ => :energy,
    :lp__ => :lp,
    :n_leapfrog__ => :n_steps,
    :stepsize__ => :step_size,
    :treedepth__ => :tree_depth,
)
const stats_key_map = merge(turing_key_map, stan_key_map)

headtail(x) = x[1], x[2:end]

function split_locname(name::AbstractString)
    name = replace(name, r"[\[,]" => '.')
    name = replace(name, ']' => "")
    name, loc = headtail(split(name, '.'))
    isempty(loc) && return name, ()
    loc = tryparse.(Int, loc)
    Nothing <: eltype(loc) && return name, ()
    return name, tuple(loc...)
end
function split_locname(name::Symbol)
    subname, loc = split_locname(string(name))
    return Symbol(subname), loc
end

function varnames_locs(loc_names)
    vars_to_locs = Dict{Symbol,Any}()
    for loc_name in loc_names
        var_name, loc = split_locname(loc_name)
        if var_name ∉ keys(vars_to_locs)
            vars_to_locs[var_name] = ([loc_name], [loc])
        else
            push!(vars_to_locs[var_name][1], loc_name)
            push!(vars_to_locs[var_name][2], loc)
        end
    end
    # ensure that elements are ordered in the same order as they would be iterated
    for loc_name_locs in values(vars_to_locs)
        perm = sortperm(loc_name_locs[2]; by=CartesianIndex)
        permute!(loc_name_locs[1], perm)
        permute!(loc_name_locs[2], perm)
    end
    return NamedTuple(vars_to_locs)
end

function attributes_dict(chns::Chains)
    info = Base.structdiff(chns.info, NamedTuple{(:hashedsummary,)})
    return Dict{String,Any}((string(k), v) for (k, v) in pairs(info))
end

function section_namedtuple(chns::Chains, section)
    ndraws, _, nchains = size(chns)
    loc_names = chns.name_map[section]
    vars_to_locs = varnames_locs(loc_names)
    vars_to_arrays = map(vars_to_locs) do names_locs
        loc_names, locs = names_locs
        sizes = reduce((a, b) -> max.(a, b), locs)
        ndim = length(sizes)
        # NOTE: slicing specific entries from AxisArrays does not preserve order
        # https://github.com/JuliaArrays/AxisArrays.jl/issues/182
        oldarr = replacemissing(permutedims(chns.value[:, loc_names, :], (3, 1, 2)))
        if iszero(ndim)
            arr = dropdims(oldarr; dims=3)
        else
            arr = Array{Union{typeof(NaN),eltype(oldarr)}}(undef, nchains, ndraws, sizes...)
            fill!(arr, NaN)
            for i in eachindex(locs, loc_names)
                @views arr[:, :, locs[i]...] = oldarr[:, :, loc_names[i]]
            end
        end
        return arr
    end
    return vars_to_arrays
end

function chains_to_namedtuple(
    chns::Chains; ignore=(), section=:parameters, rekey_fun=identity
)
    section in sections(chns) || return (;)
    chns_data = section_namedtuple(chns, section)
    chns_data_return = NamedTuple{filter(∉(ignore), keys(chns_data))}(chns_data)
    return rekey_fun(chns_data_return)
end

"""
    convert_to_inference_data(obj::Chains; group = :posterior, kwargs...) -> InferenceData

Convert the chains `obj` to an [`InferenceData`](@ref) with the specified `group`.

Remaining `kwargs` are forwarded to [`from_mcmcchains`](@ref).
"""
function convert_to_inference_data(chns::Chains; group::Symbol=:posterior, kwargs...)
    group === :posterior && return from_mcmcchains(chns; kwargs...)
    return from_mcmcchains(; group => chns, kwargs...)
end

@doc doc"""
    from_mcmcchains(posterior::MCMCChains.Chains; kwargs...) -> InferenceData
    from_mcmcchains(; kwargs...) -> InferenceData
    from_mcmcchains(
        posterior::MCMCChains.Chains,
        posterior_predictive::Any,
        predictions::Any,
        log_likelihood::Any;
        kwargs...
    ) -> InferenceData

Convert data in an `MCMCChains.Chains` format into an [`InferenceData`](@ref).

Any keyword argument below without an an explicitly annotated type above is allowed, so long
as it can be passed to [`convert_to_inference_data`](@ref).

# Arguments

- `posterior::MCMCChains.Chains`: Draws from the posterior

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
- `coords::Dict{String,Vector}=Dict()`: Map from named dimension to named indices
- `dims::Dict{String,Vector{String}}=Dict()`: Map from variable name to names of its
    dimensions
- `eltypes::Dict{String,DataType}=Dict()`: Apply eltypes to specific variables. This is used
    to assign discrete eltypes to discrete variables.

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
    eltypes=(;),
    kwargs...,
)
    rekey_fun = d -> rekey(d, stats_key_map)

    # Convert chains to dicts
    if posterior === nothing
        post_data = nothing
        stats_data = nothing
    else
        post_data = convert_to_eltypes(chains_to_namedtuple(posterior), eltypes)
        stats_data = chains_to_namedtuple(posterior; section=:internals, rekey_fun)
        stats_data = enforce_stat_eltypes(stats_data)
        stats_data = convert_to_eltypes(stats_data, (; is_accept=Bool))
    end

    all_idata = InferenceData()
    for (group, group_data) in [
        :posterior_predictive => posterior_predictive,
        :predictions => predictions,
        :log_likelihood => log_likelihood,
    ]
        group_data === nothing && continue
        if group_data isa Symbol
            group_data = (group_data,)
        end
        if Base.isiterable(typeof(group_data)) && all(Base.Fix2(isa, Symbol), group_data)
            group_data = NamedTuple{Tuple(group_data)}(post_data)
            post_data = NamedTuple{Tuple(setdiff(keys(post_data), keys(group_data)))}(
                post_data
            )
        end
        group_dataset = if group_data isa Chains
            convert_to_dataset(group_data; library, eltypes, kwargs...)
        else
            convert_to_dataset(group_data; library, kwargs...)
        end
        all_idata = merge(all_idata, InferenceData(; group => group_dataset))
    end
    post_idata = from_namedtuple(post_data; sample_stats=stats_data, library, kwargs...)
    all_idata = merge(all_idata, post_idata)
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
    eltypes=(;),
    kwargs...,
)
    all_idata = from_mcmcchains(
        posterior,
        posterior_predictive,
        predictions,
        log_likelihood;
        library,
        eltypes,
        kwargs...,
    )

    if prior !== nothing
        pre_prior_idata = convert_to_inference_data(
            prior; posterior_predictive=prior_predictive, library, eltypes, kwargs...
        )
        prior_idata = rekey(
            pre_prior_idata,
            (
                posterior=:prior,
                posterior_predictive=:prior_predictive,
                sample_stats=:sample_stats_prior,
            ),
        )
        all_idata = merge(all_idata, prior_idata)
    elseif prior_predictive !== nothing
        if prior_predictive isa Chains
            pre_prior_predictive_idata = convert_to_inference_data(
                prior_predictive; library, eltypes, kwargs...
            )
        else
            pre_prior_predictive_idata = convert_to_inference_data(
                prior_predictive; library, kwargs...
            )
        end
        all_idata = merge(
            all_idata,
            InferenceData(; prior_predictive=pre_prior_predictive_idata.posterior),
        )
    end

    for (group, group_data) in [
        :observed_data => observed_data,
        :constant_data => constant_data,
        :predictions_constant_data => predictions_constant_data,
    ]
        group_data === nothing && continue
        group_data = convert_to_eltypes(group_data, eltypes)
        group_dataset = convert_to_dataset(group_data; library, default_dims=(), kwargs...)
        all_idata = merge(all_idata, InferenceData(; group => group_dataset))
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
