import .MCMCChains: AbstractChains, sections

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
    info = delete(chns.info, :hashedsummary)
    return Dict{String,Any}((string(k), v) for (k, v) in pairs(info))
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
    from_mcmcchains(
        posterior::AbstractChains,
        posterior_predictive::Any,
        log_likelihood::String;
        kwargs...
    ) -> InferenceData

Convert data in an `MCMCChains.AbstractChains` format into an
[`InferenceData`](@ref).

Any keyword argument below without an an explicitly annotated type above is
allowed, so long as it can be passed to [`convert_to_inference_data`](@ref).

# Arguments
- `posterior::AbstractChains`: Draws from the posterior

# Keywords
- `posterior_predictive::Any=nothing`: Draws from the posterior predictive
     distribution or name(s) of predictive variables in `posterior`
- `prior::Any=nothing`: Draws from the prior
- `prior_predictive::Any=nothing`: Draws from the prior predictive distribution
     or name(s) of predictive variables in `prior`
- `observed_data::Dict{String,Array}=nothing`: Observed data on which the
     `posterior` is conditional. It should only contain data which is modeled as
     a random variable. Keys are parameter names and values.
- `constant_data::Dict{String,Array}=nothing`: Model constants, data included
     in the model which is not modeled as a random variable. Keys are parameter
     names and values.
- `log_likelihood::String=nothing`: Name of variable in `posterior` with log
     likelihoods
- `library=MCMCChains`: Name of library that generated the chains
- `coords::Dict{String,Vector}=nothing`: Map from named dimension to named
     indices
- `dims::Dict{String,Vector{String}}=nothing`: Map from variable name to names
     of its dimensions

# Returns
- `InferenceData`: The data with groups corresponding to the provided data
"""
function from_mcmcchains(
    posterior,
    posterior_predictive,
    log_likelihood;
    library = MCMCChains,
    kwargs...,
)
    kwargs = convert(Dict, merge((; dims = nothing), kwargs))
    rekey_fun = d -> rekey(d, stats_key_map)

    # Convert chains to dicts
    post_dict = chains_to_dict(posterior)
    stats_dict = chains_to_dict(posterior; section = :internals, rekey_fun = rekey_fun)
    stats_dict = enforce_stat_types(stats_dict)

    if typeof(posterior_predictive) <: Union{String,Vector{String}}
        post_pred_idata = InferenceData()
        post_pred_dict = popsubdict!(post_dict, posterior_predictive)
    else
        if posterior_predictive !== nothing
            post_pred_dataset = convert_to_dataset(
                posterior_predictive;
                library = library,
                kwargs...,
            )
            post_pred_idata = InferenceData(posterior_predictive = post_pred_dataset)
        else
            post_pred_idata = InferenceData()
        end
        post_pred_dict = nothing
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

    attrs = attributes_dict(posterior)
    attrs = merge(attrs, Dict("inference_library" => string(library)))
    kwargs = convert(Dict, merge((; attrs = attrs, dims = nothing), kwargs))
    post_idata = _from_dict(
        post_dict;
        sample_stats = stats_dict,
        posterior_predictive = post_pred_dict,
        kwargs...,
    )

    idata = InferenceData(; groups(post_idata)..., groups(post_pred_idata)...)
    return idata
end

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
    kwargs = convert(Dict, merge((; dims = nothing, coords = nothing), kwargs))

    all_idata = InferenceData[]
    post_idata = from_mcmcchains(
        posterior,
        posterior_predictive,
        log_likelihood;
        library = library,
        kwargs...,
    )
    push!(all_idata, post_idata)

    if prior !== nothing
        pre_prior_idata = convert_to_inference_data(
            prior;
            posterior_predictive = prior_predictive,
            library = library,
            kwargs...,
        )
        prior_idata = rekey(
            pre_prior_idata,
            Dict(
                :posterior => :prior,
                :posterior_predictive => :prior_predictive,
                :sample_stats => :sample_stats_prior,
            ),
        )
        push!(all_idata, prior_idata)
    end

    if observed_data !== nothing
        observed_idata = InferenceData(observed_data = convert_to_constant_dataset(
            observed_data;
            dims = kwargs[:dims],
            coords = kwargs[:coords],
        ))
        push!(all_idata, observed_idata)
    end

    if constant_data !== nothing
        constant_idata = InferenceData(constant_data = convert_to_constant_dataset(
            constant_data;
            dims = kwargs[:dims],
            coords = kwargs[:coords],
        ))
        push!(all_idata, constant_idata)
    end

    all_groups = mapreduce(groups, merge, all_idata)
    idata = InferenceData(; all_groups...)
    return idata
end

"""
    from_cmdstan(posterior::AbstractChains; kwargs...) -> InferenceData

Call [`from_mcmcchains`](@ref) on output of `CmdStan`.
"""
from_cmdstan(posterior::AbstractChains; kwargs...) =
    from_mcmcchains(posterior; library = "CmdStan", kwargs...)
