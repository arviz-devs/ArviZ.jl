@doc doc"""
    from_turing([posterior::Chains]; kwargs...) -> InferenceData

Convert data from Turing into an [`InferenceData`](@ref).

This permits passing a Turing `Model` and a random number generator to
`model` and `rng` keywords to automatically generate groups. By default,
if `posterior`, `observed_data`, and `model` are provided, then the 
`prior`, `prior_predictive`, `posterior_predictive`, and `log_likelihood`
groups are automatically generated. To avoid generating a group, provide
group data or set it to `false`.

# Arguments

- `posterior::Chains`: Draws from the posterior

# Keywords

- `model::Turing.DynamicPPL.Model`: A Turing model conditioned on observed and
constant data. `constant_data` must be provided for the model to be used.
- `rng::AbstractRNG=Random.default_rng()`: a random number generator used for
sampling from the prior, posterior predictive and prior predictive
distributions.
- `nchains::Int`: Number of chains for prior samples, defaulting to the number
of chains in the posterior, if provided, else 1.
- `ndraws::Int`: Number of draws per chain for prior samples, defaulting to the
number of draws per chain in the posterior, if provided, else 1,000.
- `kwargs`: For remaining keywords, see [`from_mcmcchains`](@ref).

# Examples

```jldoctest
julia> using Turing, Random

julia> rng = Random.seed!(42);

julia> @model function demo(xs, y, n=length(xs))
           s ~ InverseGamma(2, 3)
           m ~ Normal(0, √s)
           for i in 1:n
               xs[i] ~ Normal(m, √s)
           end
           y ~ Normal(m, √s)
       end;

julia> observed_data = (xs=[0.87, 0.08, 0.53], y=-0.85);

julia> model = demo(observed_data...);

julia> chn = sample(rng, model, NUTS(), 1_000; progress=false);

julia> from_turing(chn; model=model, rng=rng, observed_data=observed_data, prior=false)
InferenceData with groups:
	> posterior
	> posterior_predictive
	> log_likelihood
	> sample_stats
	> observed_data
	> constant_data
```
"""
function from_turing(
    chns=nothing;
    model::Union{Nothing,Turing.DynamicPPL.Model}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    nchains=ndraws = chns isa Turing.MCMCChains.Chains ? last(size(chns)) : 1,
    ndraws=chns isa Turing.MCMCChains.Chains ? first(size(chns)) : 1_000,
    observed_data=true,
    constant_data=true,
    posterior_predictive=true,
    prior=true,
    prior_predictive=true,
    log_likelihood=true,
    kwargs...,
)
    kwargs = Dict{Symbol,Any}(kwargs)
    kwargs[:library] = Turing

    groups = Dict{Symbol,Any}(
        :observed_data => observed_data,
        :constant_data => constant_data,
        :posterior_predictive => posterior_predictive,
        :prior => prior,
        :prior_predictive => prior_predictive,
        :log_likelihood => log_likelihood,
    )
    groups_to_generate = Set(k for (k, v) in groups if v === true)
    for (name, value) in groups
        if value isa Bool
            groups[name] = nothing
        end
    end

    model === nothing && return from_mcmcchains(chns; groups..., kwargs...)
    if :prior in groups_to_generate
        groups[:prior] = _sample_prior(rng, model, nchains, ndraws)
    end

    if groups[:observed_data] === nothing
        return from_mcmcchains(chns; groups..., kwargs...)
    end

    observed_data = groups[:observed_data]
    observed_data_keys = Set(
        observed_data isa Dict ? Symbol.(keys(observed_data)) : propertynames(observed_data)
    )

    if :constant_data in groups_to_generate
        groups[:constant_data] = Dict(
            filter(p -> first(p) ∉ observed_data_keys, pairs(model.args))
        )
    end

    if :prior_predictive in groups_to_generate
        if groups[:prior] isa Turing.MCMCChains.Chains
            groups[:prior_predictive] = _sample_predictive(
                rng, model, groups[:prior], observed_data_keys
            )
        elseif groups[:prior] !== nothing
            @warn "Could not generate group :prior_predictive because group :prior was not an MCMCChains.Chains."
        end
    end

    if :posterior_predictive in groups_to_generate
        if chns isa Turing.MCMCChains.Chains
            groups[:posterior_predictive] = _sample_predictive(
                rng, model, chns, observed_data_keys
            )
        elseif chns !== nothing
            @warn "Could not generate group :posterior_predictive because group :posterior was not an MCMCChains.Chains."
        end
    end

    if :log_likelihood in groups_to_generate
        if chns isa Turing.MCMCChains.Chains
            groups[:log_likelihood] = _compute_log_likelihood(model, chns)
        elseif chns !== nothing
            @warn "Could not generate log_likelihood because posterior must be an MCMCChains.Chains."
        end
    end

    idata = from_mcmcchains(chns; groups..., kwargs...)

    # add model name to generated InferenceData groups
    setattribute!(idata, :model_name, nameof(model))
    return idata
end

function _sample_prior(rng::AbstractRNG, model::Turing.DynamicPPL.Model, nchains, ndraws)
    return Turing.sample(
        rng, model, Turing.Prior(), Turing.MCMCThreads(), ndraws, nchains; progress=false
    )
end

function _build_predictive_model(model::Turing.DynamicPPL.Model, data_keys)
    var_names = filter(in(data_keys), keys(model.args))
    return Turing.DynamicPPL.Model{var_names}(
        model.name, model.f, model.args, model.defaults
    )
end

function _sample_predictive(
    rng::AbstractRNG, model::Turing.DynamicPPL.Model, chns, data_keys
)
    model_predict = _build_predictive_model(model, data_keys)
    return Turing.predict(rng, model_predict, chns)
end

function _compute_log_likelihood(
    model::Turing.DynamicPPL.Model, chns::Turing.MCMCChains.Chains
)
    chains_only_params = Turing.MCMCChains.get_sections(chns, :parameters)
    loglikelihoods = Turing.pointwise_loglikelihoods(model, chains_only_params)
    pred_names = sort(collect(keys(loglikelihoods)); by=split_locname)
    loglikelihoods_vals = getindex.(Ref(loglikelihoods), pred_names)
    # Bundle loglikelihoods into a `Chains` object so we can reuse our own variable
    # name parsing
    loglikelihoods_arr = permutedims(cat(loglikelihoods_vals...; dims=3), (1, 3, 2))
    return Turing.MCMCChains.Chains(loglikelihoods_arr, pred_names)
end
