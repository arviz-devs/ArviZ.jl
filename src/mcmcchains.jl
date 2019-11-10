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
    reshape_values(x::AbstractArray)

Convert from `MCMCChains` parameter values with dimensions
`(ndraw, size..., nchain)` to ArviZ's expected `(nchain, ndraw, size...)`.
"""
reshape_values(x::AbstractArray{T,N}) where {T,N} = permutedims(x, (N, 1, 2:(N-1)...))

headtail(x) = x[1], x[2:end]

function split_locname(name)
    name = replace(name, r"[\[,]" => '.')
    name = replace(name, ']' => "")
    name, loc = headtail(split(name, '.'))
    length(loc) == 0 && return name, ()
    loc = tryparse.(Int, loc)
    Nothing <: eltype(loc) && return name, ()
    return name, tuple(loc...)
end

function varnames_locs_dict(loc_names)
    vars_to_locs = Dict()
    for loc_name in loc_names
        var_name, loc = split_locname(loc_name)
        if var_name ∉ keys(vars_to_locs)
            vars_to_locs[var_name] = ([loc_name], [loc])
        else
            push!(vars_to_locs[var_name][1], loc_name)
            push!(vars_to_locs[var_name][2], loc)
        end
    end
    return vars_to_locs
end

function attributes_dict(chns::AbstractChains)
    info = chns.info
    :hashedsummary in propertynames(info) || return info
    chndfs = info.hashedsummary.x[2]
    names = tuple(snakecase.(getproperty.(chndfs, :name))...)
    dfs = tuple(topandas.(chndfs)...)
    info = delete(info, :hashedsummary)
    attrs = merge(info, (mcmcchains_summary = Dict(zip(names, dfs)),))
    return Dict{String,Any}((string(k), v) for (k, v) in pairs(attrs))
end

attributes_dict(::Nothing) = Dict()

function section_dict(chns::AbstractChains, section)
    ndraws, _, nchains = size(chns)
    loc_names = string.(getfield(chns.name_map, section))
    vars_to_locs = varnames_locs_dict(loc_names)
    vars_to_arrays = Dict{String,Array}()
    for (var_name, names_locs) in vars_to_locs
        loc_names, locs = names_locs
        max_loc = maximum(hcat([[loc...] for loc in locs]...); dims = 2)
        ndim = length(max_loc)
        sizes = tuple(max_loc...)

        oldarr = reshape_values(replacemissing(Array(chns.value[:, loc_names, :])))
        if ndim == 0
            arr = dropdims(oldarr; dims = 3)
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

function convert_to_dataset(chns::AbstractChains; library = MCMCChains, kwargs...)
    chns_dict = chains_to_dict(chns)
    attrs = attributes_dict(chns)
    attrs = merge(attrs, Dict("inference_library" => string(library)))
    return dict_to_dataset(chns_dict; attrs = attrs, kwargs...)
end

function convert_to_inference_data(obj::AbstractChains; group = :posterior, kwargs...)
    ds = convert_to_dataset(obj; kwargs...)
    return InferenceData(; Symbol(group) => ds)
end

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
    post_dict = chains_to_dict(posterior)
    stats_dict = chains_to_dict(
        posterior;
        section = :internals,
        rekey_fun = d -> rekey(d, stats_key_map),
    )
    stats_dict = enforce_stat_types(stats_dict)

    prior_dict = chains_to_dict(prior)
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

    if log_like_dict !== nothing && stats_dict !== nothing
        stats_dict = merge(
            stats_dict,
            Dict("log_likelihood" => log_like_dict[log_likelihood]),
        )
        if dims !== nothing && log_likelihood in keys(dims)
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
