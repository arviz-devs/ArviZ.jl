const stats_key_map = Dict(
    "acceptance_rate" => "mean_tree_accept",
    "hamiltonian_energy" => "energy",
    "n_steps" => "tree_size",
    "numerical_error" => "diverging",
    "tree_depth" => "depth",
)

missingtonan(x) = replace(x, missing => NaN)

reshape_values(x) = permutedims(x, [2, 1])
reshape_values(x::NTuple) = permutedims(cat(x...; dims = 3), [2, 1, 3])

function rekey(d, keymap)
    dnew = empty(d)
    for (k, v) in d
        knew = get(keymap, k, k)
        haskey(dnew, knew) && throw(ArgumentError("$knew in `keymap` is already in `d`."))
        dnew[knew] = d[k]
    end
    return dnew
end

function section_dict(chn, section)
    params = get(chn, section = section; flatten = false)
    names = string.(keys(params))
    vals = missingtonan.(reshape_values.(values(params)))
    return Dict(zip(names, vals))
end

section_dict(::Nothing, section) = nothing

function sample_stats_dict(chn)
    section = :internals
    in(section, keys(chn.name_map)) || return nothing
    stats = section_dict(chn, section)
    return rekey(stats, stats_key_map)
end

sample_stats_dict(::Nothing) = nothing

"""
    from_mcmcchains(
        posterior::Union{AbstractChains,Nothing} = nothing;
        prior::Union{AbstractChains,Nothing} = nothing,
        kwargs...,
    )

Convert the chains in `posterior` and/or `prior` to `InferenceData`. The
chains are assigned to the corresponding groups. Remaining `kwargs` are
forwarded to `from_dict`.
"""
function from_mcmcchains(
    posterior::Union{AbstractChains,Nothing} = nothing;
    prior::Union{AbstractChains,Nothing} = nothing,
    kwargs...,
)
    post_dict = section_dict(posterior, :parameters)
    sample_stats = sample_stats_dict(posterior)
    prior_dict = section_dict(prior, :parameters)
    sample_stats_prior = sample_stats_dict(prior)
    return from_dict(
        posterior = post_dict;
        sample_stats = sample_stats,
        prior = prior_dict,
        sample_stats_prior = sample_stats_prior,
        kwargs...,
    )
end

function convert_to_inference_data(obj::AbstractChains; kwargs...)
    return from_mcmcchains(obj; kwargs...)
end
