using .MCMCChains: AbstractChains, ChainDataFrame, sections
using .DataFrames

export from_mcmcchains

const turing_key_map = Dict(
    "acceptance_rate" => "mean_tree_accept",
    "hamiltonian_energy" => "energy",
    "n_steps" => "tree_size",
    "numerical_error" => "diverging",
    "tree_depth" => "depth",
)
const stan_key_map = Dict(
    "accept_stat__" => "accept_stat",
    "divergent__" => "diverging",
    "energy__" => "energy",
    "lp__" => "lp",
    "n_leapfrog__" => "n_leapfrog",
    "stepsize__" => "stepsize",
    "treedepth__" => "treedepth",
)
const stats_key_map = merge(turing_key_map, stan_key_map)

function topandas(df::DataFrames.DataFrame)
    cols = replacemissing.(eachcol(df))
    colnames = names(df)
    df = DataFrames.DataFrame(cols, colnames)
    return Pandas.DataFrame(df)
end

topandas(df::ChainDataFrame) = topandas(df.df)

"""
    reshape_values(x::AbstractMatrix)

Convert from MCMCChains' parameter values with dimensions `(ndraw, nchain)` to
ArviZ's expected `(nchain, ndraw)`.
"""
reshape_values(x::AbstractMatrix) = permutedims(x, [2, 1])

"""
    reshape_values(x::NTuple)

Given `nparam` parameter vectors, each for an index of a multivariate
parameter, each with dimensions `(ndraw, nchain)`, convert to an array with
ArviZ's  expected `(nchain, ndraw, nparam)`.
"""
reshape_values(x::NTuple) = cat(reshape_values.(x)...; dims = 3)

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
    vals = replacemissing.(reshape_values.(values(params)))
    return Dict(zip(names, vals))
end

function info_namedtuple(chn)
    info = chn.info
    :hashedsummary in propertynames(info) || return info
    chndfs = info.hashedsummary.x[2]
    names = tuple(getproperty.(chndfs, :name)...)
    dfs = tuple(topandas.(getproperty.(chndfs, :df))...)
    info = merge(delete(info, :hashedsummary), namedtuple(names, dfs))
    return info
end

function chains_to_dataset(
    chn::AbstractChains;
    section = :parameters,
    library = MCMCChains,
    rekey_fun = identity,
    kwargs...,
)
    chn_dict = section_dict(chn, section)
    chn_dict = rekey_fun(chn_dict)
    attrs = merge(info_namedtuple(chn), (inference_library = string(library),))
    attrs = convert(Dict, attrs)
    return dict_to_dataset(chn_dict; attrs = attrs, kwargs...)
end

function chains_to_stats_dataset(
    chn::AbstractChains;
    rekey_fun = d -> rekey(d, stats_key_map),
    kwargs...,
)
    :internals in sections(chn) || return nothing
    return chains_to_dataset(chn; section = :internals, rekey_fun = rekey_fun, kwargs...)
end

function convert_to_dataset(chn::AbstractChains; kwargs...)
    return chains_to_dataset(chn; kwargs...)
end

"""
    group_to_dataset(group, sample; kwargs...)

Convert `group` directly to dataset if possible. If not, assume `group`
contains identifiers from `sample` and extract. `kwargs` are passed to
`convert_to_dataset`. Returns dataset and potentially modified `sample`.
"""
function group_to_dataset(group, sample; kwargs...)
    return convert_to_dataset(group; kwargs...), sample
end

group_to_dataset(group::String, sample; kwargs...) =
    group_to_dataset([group], sample; kwargs...)
group_to_dataset(group::Vector{String}, sample; kwargs...) =
    convert_to_dataset(sample[group]; kwargs...), sample
group_to_dataset(::Nothing, sample; kwargs...) = nothing, sample

function from_mcmcchains(
    posterior = nothing;
    posterior_predictive = nothing,
    prior = nothing,
    prior_predictive = nothing,
    observed_data = nothing,
    constant_data = nothing,
    log_likelihood = nothing,
    kwargs...,
)
    postpred_data, posterior = group_to_dataset(posterior_predictive, posterior)
    obs_data, posterior = group_to_dataset(observed_data, posterior)
    const_data, posterior = group_to_dataset(constant_data, posterior)
    # log_like = group_to_dataset(log_likelihood, posterior)
    priorpred_data, prior = group_to_dataset(prior_predictive, prior)

    if !isnothing(posterior)
        post_data = chains_to_dataset(posterior; kwargs...)
        stats_data = chains_to_stats_dataset(posterior; kwargs...)
    else
        post_data, stats_data = nothing, nothing
    end

    if !isnothing(prior)
        prior_data = chains_to_dataset(prior; section = :parameters, kwargs...)
        prior_stats_data = chains_to_stats_dataset(posterior; kwargs...)
    else
        prior_data, prior_stats_data = nothing, nothing
    end

    return InferenceData(
        ;
        posterior = post_data,
        sample_stats = stats_data,
        posterior_predictive = postpred_data,
        prior = prior_data,
        sample_stats_prior = prior_stats_data,
        prior_predictive = priorpred_data,
        observed_data = obs_data,
        constant_data = const_data,
    )
end

from_cmdstan(data::AbstractChains; kwargs...) = from_mcmcchains(data; kwargs...)

function convert_to_inference_data(obj::AbstractChains; kwargs...)
    return from_mcmcchains(obj; kwargs...)
end
