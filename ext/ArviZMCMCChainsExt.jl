module ArviZMCMCChainsExt

if isdefined(Base, :get_extension)
    using ArviZ: ArviZ, InferenceObjects
    using MCMCChains: MCMCChains
else
    using ..ArviZ: ArviZ, InferenceObjects
    using ..MCMCChains: MCMCChains
end

const stats_key_map = Dict(
    :hamiltonian_energy => :energy,
    :hamiltonian_energy_error => :energy_error,
    :is_adapt => :tune,
    :max_hamiltonian_energy_error => :max_energy_error,
    :nom_step_size => :step_size_nom,
    :numerical_error => :diverging,
)

headtail(x) = x[1], x[2:end]

function split_locname(name::AbstractString)
    endswith(name, "]") || return name, ()
    basename, index = rsplit(name[1:end-1], "["; limit=2)
    isempty(index) && return name, ()
    try
        loc = parse.(Int, split(index, ','))
        return basename, tuple(loc...)
    catch e
        e isa ArgumentError && return name, ()
        rethrow(e)
    end
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

function attributes_dict(chns::MCMCChains.Chains)
    info = Base.structdiff(chns.info, NamedTuple{(:hashedsummary,)})
    return Dict{String,Any}((string(k), v) for (k, v) in pairs(info))
end

function section_namedtuple(chns::MCMCChains.Chains, section)
    ndraws, _, nchains = size(chns)
    loc_names = chns.name_map[section]
    vars_to_locs = varnames_locs(loc_names)
    vars_to_arrays = map(vars_to_locs) do names_locs
        loc_names, locs = names_locs
        sizes = reduce((a, b) -> max.(a, b), locs)
        ndim = length(sizes)
        # NOTE: slicing specific entries from AxisArrays does not preserve order
        # https://github.com/JuliaArrays/AxisArrays.jl/issues/182
        oldarr = ArviZ.replacemissing(permutedims(chns.value[:, loc_names, :], (1, 3, 2)))
        if iszero(ndim)
            arr = dropdims(oldarr; dims=3)
        else
            arr = Array{Union{typeof(NaN),eltype(oldarr)}}(undef, ndraws, nchains, sizes...)
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
    chns::MCMCChains.MCMCChains.Chains; ignore=(), section=:parameters, rekey_fun=identity
)
    section in MCMCChains.sections(chns) || return (;)
    chns_data = section_namedtuple(chns, section)
    chns_data_return = NamedTuple{filter(∉(ignore), keys(chns_data))}(chns_data)
    return rekey_fun(chns_data_return)
end

"""
    convert_to_inference_data(obj::MCMCChains.Chains; group = :posterior, kwargs...) -> InferenceData

Convert the chains `obj` to an [`InferenceData`](@ref) with the specified `group`.

Remaining `kwargs` are forwarded to [`from_mcmcchains`](@ref).
"""
function InferenceObjects.convert_to_inference_data(
    chns::MCMCChains.Chains; group::Symbol=:posterior, kwargs...
)
    group === :posterior && return ArviZ.from_mcmcchains(chns; kwargs...)
    return ArviZ.from_mcmcchains(; group => chns, kwargs...)
end

function ArviZ.from_mcmcchains(
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
        post_data = ArviZ.convert_to_eltypes(chains_to_namedtuple(posterior), eltypes)
        stats_data = chains_to_namedtuple(posterior; section=:internals, rekey_fun)
        stats_data = ArviZ.enforce_stat_eltypes(stats_data)
        stats_data = ArviZ.convert_to_eltypes(stats_data, (; is_accept=Bool))
    end

    all_idata = InferenceObjects.InferenceData()
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
        group_dataset = if group_data isa MCMCChains.Chains
            InferenceObjects.convert_to_dataset(group_data; library, eltypes, kwargs...)
        else
            InferenceObjects.convert_to_dataset(group_data; library, kwargs...)
        end
        all_idata = merge(
            all_idata, InferenceObjects.InferenceData(; group => group_dataset)
        )
    end
    post_idata = ArviZ.from_namedtuple(
        post_data; sample_stats=stats_data, library, kwargs...
    )
    all_idata = merge(all_idata, post_idata)
    return all_idata
end
function ArviZ.from_mcmcchains(
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
    all_idata = ArviZ.from_mcmcchains(
        posterior,
        posterior_predictive,
        predictions,
        log_likelihood;
        library,
        eltypes,
        kwargs...,
    )

    if prior !== nothing
        pre_prior_idata = InferenceObjects.convert_to_inference_data(
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
        if prior_predictive isa MCMCChains.Chains
            pre_prior_predictive_idata = InferenceObjects.convert_to_inference_data(
                prior_predictive; library, eltypes, kwargs...
            )
        else
            pre_prior_predictive_idata = InferenceObjects.convert_to_inference_data(
                prior_predictive; library, kwargs...
            )
        end
        all_idata = merge(
            all_idata,
            InferenceObjects.InferenceData(;
                prior_predictive=pre_prior_predictive_idata.posterior
            ),
        )
    end

    for (group, group_data) in [
        :observed_data => observed_data,
        :constant_data => constant_data,
        :predictions_constant_data => predictions_constant_data,
    ]
        group_data === nothing && continue
        group_data = ArviZ.convert_to_eltypes(group_data, eltypes)
        group_dataset = ArviZ.convert_to_dataset(
            group_data; library, default_dims=(), kwargs...
        )
        all_idata = merge(
            all_idata, InferenceObjects.InferenceData(; group => group_dataset)
        )
    end

    return all_idata
end

# adapted from InferenceObjects.jl
rekey(d, keymap) = Dict(get(keymap, k, k) => d[k] for k in keys(d))
function rekey(d::NamedTuple, keymap)
    new_keys = map(k -> get(keymap, k, k), keys(d))
    return NamedTuple{new_keys}(values(d))
end
function rekey(data::InferenceObjects.InferenceData, keymap)
    groups_old = InferenceObjects.groups(data)
    names_new = map(k -> get(keymap, k, k), propertynames(groups_old))
    groups_new = NamedTuple{names_new}(Tuple(groups_old))
    return InferenceObjects.InferenceData(groups_new)
end

end  # module
