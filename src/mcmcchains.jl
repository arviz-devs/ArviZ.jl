const stats_key_map = Dict(
    "acceptance_rate" => "accept_stat",
    "step_size" => "stepsize",
    "tree_depth" => "treedepth",
    "n_steps" => "n_leapfrog",
    "numerical_error" => "diverging",
    "hamiltonian_energy" => "energy",
)

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
    vals = reshape_values.(values(params))
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

# Just support basic interface for now.
function from_mcmcchains(posterior = nothing, args...; prior = nothing, kwargs...)
    data_post = section_dict(posterior, :parameters)
    sample_stats = sample_stats_dict(posterior)
    data_prior = section_dict(prior, :parameters)
    sample_stats_prior = sample_stats_dict(prior)
    return from_dict(
        posterior = data_post,
        args...;
        prior = data_prior,
        sample_stats = sample_stats,
        sample_stats_prior = sample_stats_prior,
        kwargs...,
    )
end

function convert_to_inference_data(chn::AbstractChains; kwargs...)
    return from_mcmcchains(chn; kwargs...)
end
