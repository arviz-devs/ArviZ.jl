using .MCMCChains: AbstractChains, ChainDataFrame, sections
using .DataFrames

export from_mcmcchains

const turing_key_map = Dict(
    "acceptance_rate" => "mean_tree_accept",
    "hamiltonian_energy" => "energy",
    "hamiltonian_energy_error" => "energy_error",
    "is_adapt" => "tune",
    "max_hamiltonian_energy_error" => "max_energy_error",
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
    pdf = Pandas.DataFrame(df)
    return pdf[colnames]
end

topandas(df::ChainDataFrame) = topandas(df.df)

"""
    reshape_values(x::AbstractMatrix)

Convert from MCMCChains' parameter values with dimensions `(ndraw, nchain)` to
ArviZ's expected `(nchain, ndraw)`.
"""
reshape_values(x::AbstractMatrix) = permutedims(x, [2, 1])
reshape_values(x::AbstractArray{T,3}) where {T} = permutedims(x, [3, 1, 2])

"""
    reshape_values(x::NTuple)

Given `nparam` parameter vectors, each for an index of a multivariate
parameter, each with dimensions `(ndraw, nchain)`, convert to an array with
ArviZ's  expected `(nchain, ndraw, nparam)`.
"""
reshape_values(x::NTuple) = cat(reshape_values.(x)...; dims = 3)

function attributes_dict(chns::AbstractChains)
    info = chns.info
    :hashedsummary in propertynames(info) || return info
    chndfs = info.hashedsummary.x[2]
    names = tuple(snakecase.(getproperty.(chndfs, :name))...)
    dfs = tuple(topandas.(getproperty.(chndfs, :df))...)
    info = delete(info, :hashedsummary)
    attrs = merge(info, (mcmcchains_summary = Dict(zip(names, dfs)),))
    return Dict{String,Any}((string(k), v) for (k, v) in pairs(attrs))
end

attributes_dict(::Nothing) = Dict()

function section_dict(chns::AbstractChains, section)
    params = get(chns, section = section; flatten = false)
    names = string.(keys(params))
    vals = replacemissing.(reshape_values.(values(params)))
    return Dict(zip(names, vals))
end

function chains_to_dict(
    chns::AbstractChains;
    ignore = String[],
    section = :parameters,
    rekey_fun = identity,
)
    section in sections(chns) || return Dict()
    chns_dict = section_dict(chns, section)
    removekeys!(chns_dict, ignore)
    return rekey_fun(chns_dict)
end

chains_to_dict(::Nothing; kwargs...) = nothing

function chains_to_dataset(
    chns::AbstractChains;
    ignore = String[],
    section = :parameters,
    library = MCMCChains,
    rekey_fun = identity,
    kwargs...,
)
    chns_dict = chains_to_dict(
        chns;
        ignore = ignore,
        section = section,
        rekey_fun = rekey_fun,
    )
    attrs = attributes_dict(chns; library = library)
    return dict_to_dataset(chns_dict; attrs = attrs, kwargs...)
end

function convert_to_dataset(chns::AbstractChains; kwargs...)
    return chains_to_dataset(chns; kwargs...)
end

function indexify_chains(chns)
    params = MCMCChains.names(chns)
    d = indexify(params)
    length(d) == 0 && return chns
    return MCMCChains.set_names(chns, d)
end

indexify_chains(::Nothing) = nothing

function from_mcmcchains(
    posterior = nothing;
    posterior_predictive = nothing,
    prior = nothing,
    prior_predictive = nothing,
    observed_data = nothing,
    constant_data = nothing,
    log_likelihood::Union{String,Nothing} = nothing,
    library = MCMCChains,
    dims = nothing,
    kwargs...,
)
    post_dict = chains_to_dict(indexify_chains(posterior))
    stats_dict = chains_to_dict(
        posterior;
        section = :internals,
        rekey_fun = d -> rekey(d, stats_key_map),
    )
    stats_dict = enforce_stat_types(stats_dict)

    prior_dict = chains_to_dict(indexify_chains(prior))
    prior_stats_dict = chains_to_dict(
        prior;
        section = :internals,
        rekey_fun = d -> rekey(d, stats_key_map),
    )
    prior_stats_dict = enforce_stat_types(prior_stats_dict)

    postpred_dict = popsubdict!(post_dict, posterior_predictive)
    obs_data_dict = popsubdict!(post_dict, observed_data)
    const_data_dict = popsubdict!(post_dict, constant_data)
    log_like_dict = popsubdict!(post_dict, log_likelihood)
    priorpred_dict = popsubdict!(prior_dict, prior_predictive)

    if !isnothing(log_like_dict) && !isnothing(stats_dict)
        stats_dict = merge(
            stats_dict,
            Dict("log_likelihood" => log_like_dict[log_likelihood]),
        )
        if !isnothing(dims) && log_likelihood in keys(dims)
            dims = merge(dims, Dict("log_likelihood" => dims[log_likelihood]))
        end
    end

    attrs = attributes_dict(posterior)
    attrs = merge(attrs, Dict("inference_library" => string(library)))

    return _from_dict(
        post_dict,
        ;
        sample_stats = stats_dict,
        posterior_predictive = postpred_dict,
        prior = prior_dict,
        sample_stats_prior = prior_stats_dict,
        prior_predictive = priorpred_dict,
        observed_data = obs_data_dict,
        constant_data = const_data_dict,
        attrs = attrs,
        dims = dims,
        kwargs...,
    )
end

from_cmdstan(data::AbstractChains; kwargs...) =
    from_mcmcchains(data; library = "CmdStan", kwargs...)

function convert_to_inference_data(obj::AbstractChains; kwargs...)
    return from_mcmcchains(obj; kwargs...)
end
