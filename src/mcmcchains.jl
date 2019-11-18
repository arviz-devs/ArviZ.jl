import .MCMCChains: AbstractChains, ChainDataFrame, sections
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

"""
    topandas(df::DataFrames.DataFrame) -> Pandas.DataFrame
    topandas(df::MCMCChains.ChainDataFrame) -> Pandas.DataFrame

Convert `df` into a Pandas format, maintaining column order and replacing
`missing` with `NaN`.
"""
function topandas(df::DataFrames.DataFrame)
    cols = replacemissing.(eachcol(df))
    colnames = names(df)
    df = DataFrames.DataFrame(cols, colnames)
    pdf = Pandas.DataFrame(df)
    return pdf[colnames]
end

topandas(df::ChainDataFrame) = topandas(df.df)

"""
    reshape_values(x::AbstractArray) -> AbstractArray

Convert from `MCMCChains` variable values with dimensions
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

"""
    convert_to_dataset(chns::AbstractChains; library = MCMCChains, kwargs...) -> Dataset

Convert the chains `obj` to a [`Dataset`](@ref). `library` is the library that
created the chains. Remaining `kwargs` are forwarded to
[`dict_to_dataset`](@ref).
"""
function convert_to_dataset(chns::AbstractChains; library = MCMCChains, kwargs...)
    chns_dict = chains_to_dict(chns)
    return dict_to_dataset(chns_dict; library = library, kwargs...)
end

"""
    convert_to_inference_data(obj::AbstractChains; group = :posterior, kwargs...) -> InferenceData

Convert the chains `obj` to an [`InferenceData`](@ref) with the specified `group`.
Remaining `kwargs` are forwarded to [`from_mcmcchains`](@ref).
"""
function convert_to_inference_data(chns::AbstractChains; group = :posterior, kwargs...)
    group = Symbol(group)
    group == :posterior && return from_mcmcchains(chns; kwargs...)
    return from_mcmcchains(; group => chns)
end

"""
    from_mcmcchains(posterior::AbstractChains; kwargs...) -> InferenceData
    from_mcmcchains(; kwargs...) -> InferenceData

Convert data in an `MCMCChains.AbstractChains` format into an
[`InferenceData`](@ref).

Any keyword argument below without an an explicitly annotated type above is
allowed, so long as it can be passed to [`convert_to_inference_data`](@ref).

# Arguments
- `posterior::AbstractChains`: Draws from the posterior

# Keywords
- `posterior_predictive=nothing`: Draws from the posterior predictive distribution
     or name(s) of predictive variables in `posterior`
- `prior::AbstractChains=nothing`: Draws from the prior
- `prior_predictive=nothing`: Draws from the prior predictive distribution
     or name(s) of predictive variables in `prior`
- `observed_data=nothing`: Data actually observed or name(s) of observed data
     variables in `posterior`
- `constant_data=nothing`: Data that are constant, not observed (e.g. metadata),
     or name(s) of variables in posterior
- `log_likelihood::String=nothing`: Name of variable in `posterior` with log likelihoods
- `library=MCMCChains`: Name of library that generated the chains
- `coords::Dict{String,Vector}=nothing`: Map from named dimension to named
    indices
- `dims::Dict{String,Vector{String}}=nothing`: Map from variable name to names
    of its dimensions

# Returns
- `InferenceData`: The data with groups corresponding to the provided data
"""
function from_mcmcchains(
    posterior = nothing;
    prior = nothing,
    posterior_predictive = nothing,
    prior_predictive = nothing,
    observed_data = nothing,
    constant_data = nothing,
    log_likelihood = nothing,
    library = MCMCChains,
    kwargs...,
)
    attrs = attributes_dict(posterior)
    attrs = merge(attrs, Dict("inference_library" => string(library)))
    kwargs = convert(Dict, merge((; attrs = attrs, dims = nothing), kwargs))
    post_groups = (
        posterior_predictive = posterior_predictive,
        observed_data = observed_data,
        constant_data = constant_data,
    )

    group_dicts = Dict{Symbol,Dict}()
    group_datasets = Dict{Symbol,Dataset}()
    rekey_fun = d -> rekey(d, stats_key_map)

    # Convert chains to dicts
    post_dict = chains_to_dict(posterior)
    stats_dict = chains_to_dict(posterior; section = :internals, rekey_fun = rekey_fun)
    stats_dict = enforce_stat_types(stats_dict)

    prior_dict = chains_to_dict(prior)
    prior_stats_dict = chains_to_dict(prior; section = :internals, rekey_fun = rekey_fun)
    prior_stats_dict = enforce_stat_types(prior_stats_dict)

    # Remove variables from posterior/prior or convert to dataset
    for (group, data) in pairs(post_groups)
        if typeof(data) <: Union{String,Vector{String}}
            group_dicts[group] = popsubdict!(post_dict, data)
        elseif data !== nothing
            group_datasets[group] = convert_to_dataset(data; kwargs...)
        end
    end
    if typeof(prior_predictive) <: Union{String,Vector{String}}
        group_dicts[:prior_predictive] = popsubdict!(prior_dict, prior_predictive)
    elseif prior_predictive !== nothing
        group_datasets[:prior_predictive] = convert_to_dataset(prior_predictive; kwargs...)
    end

    # Handle log likelihood as a special case
    if log_likelihood !== nothing
        log_like_dict = popsubdict!(post_dict, log_likelihood)
        log_like_dict = Dict("log_likelihood" => log_like_dict[log_likelihood])
        if stats_dict === nothing
            stats_dict = log_like_dict
        else
            stats_dict = merge(stats_dict, log_like_dict)
        end
        dims = kwargs[:dims]
        if dims !== nothing && log_likelihood in keys(dims)
            dims["log_likelihood"] = dims[log_likelihood]
        end
    end

    idata1 = _from_dict(
        post_dict;
        sample_stats = stats_dict,
        prior = prior_dict,
        sample_stats_prior = prior_stats_dict,
        group_dicts...,
        kwargs...,
    )
    idata2 = InferenceData(; group_datasets...)
    isempty(idata1._groups) && return idata2
    isempty(idata2._groups) || concat!(idata1, idata2)
    return idata1
end

"""
    from_cmdstan(posterior::AbstractChains; kwargs...) -> InferenceData

Call [`from_mcmcchains`](@ref) on output of `CmdStan`.
"""
from_cmdstan(data::AbstractChains; kwargs...) =
    from_mcmcchains(data; library = "CmdStan", kwargs...)
