function from_turing(
    chns=nothing;
    model::Union{Nothing,Turing.DynamicPPL.Model}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    nchains=ndraws = chns isa Turing.MCMCChains.Chains ? last(size(chns)) : 1,
    ndraws=chns isa Turing.MCMCChains.Chains ? first(size(chns)) : 1_000,
    library=Turing,
    observed_data=true,
    constant_data=true,
    posterior_predictive=true,
    prior=true,
    prior_predictive=true,
    log_likelihood=true,
    kwargs...,
)
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

    model === nothing && return from_mcmcchains(chns; library=library, groups..., kwargs...)
    if :prior in groups_to_generate
        groups[:prior] = Turing.sample(
            rng,
            model,
            Turing.Prior(),
            Turing.MCMCThreads(),
            ndraws,
            nchains;
            progress=false,
        )
    end

    if groups[:observed_data] === nothing
        return from_mcmcchains(chns; library=library, groups..., kwargs...)
    end

    observed_data = groups[:observed_data]
    observed_data_keys = Set(
        observed_data isa Dict ? Symbol.(keys(observed_data)) : propertynames(observed_data)
    )

    if :constant_data in groups_to_generate
        groups[:constant_data] = NamedTuple(
            filter(p -> first(p) ∉ observed_data_keys, pairs(model.args))
        )
    end

    # Instantiate the predictive model
    data_var_names = filter(in(observed_data_keys), keys(model.args))
    model_predict = Turing.DynamicPPL.Model{data_var_names}(
        model.name, model.f, model.args, model.defaults
    )

    # and then sample!
    if :prior_predictive in groups_to_generate
        if groups[:prior] isa Turing.MCMCChains.Chains
            groups[:prior_predictive] = Turing.predict(rng, model_predict, groups[:prior])
        elseif groups[:prior] !== nothing
            @warn "Could not generate group :prior_predictive because group :prior was not an MCMCChains.Chains."
        end
    end

    if :posterior_predictive in groups_to_generate
        if chns isa Turing.MCMCChains.Chains
            groups[:posterior_predictive] = Turing.predict(rng, model_predict, chns)
        elseif chns !== nothing
            @warn "Could not generate group :posterior_predictive because group :posterior was not an MCMCChains.Chains."
        end
    end

    if :log_likelihood in groups_to_generate
        if chns isa Turing.MCMCChains.Chains
            chains_only_params = Turing.MCMCChains.get_sections(chns, :parameters)
            loglikelihoods = Turing.pointwise_loglikelihoods(model, chains_only_params)
            pred_names = sort(collect(keys(loglikelihoods)); by=split_locname)
            loglikelihoods_vals = getindex.(Ref(loglikelihoods), pred_names)
            # Bundle loglikelihoods into a `Chains` object so we can reuse our own variable
            # name parsing
            loglikelihoods_arr = permutedims(cat(loglikelihoods_vals...; dims=3), (1, 3, 2))
            groups[:log_likelihood] = Turing.MCMCChains.Chains(
                loglikelihoods_arr, pred_names
            )
        elseif chns !== nothing
            @warn "Could not generate log_likelihood because posterior must be an MCMCChains.Chains."
        end
    end

    idata = from_mcmcchains(chns; library=library, groups..., kwargs...)

    # add model name to generated InferenceData groups
    for name in groupnames(idata)
        name in (:observed_data,) && continue
        ds = getproperty(idata, name)
        setattribute!(ds, :model_name, nameof(model))
    end
    return idata
end
