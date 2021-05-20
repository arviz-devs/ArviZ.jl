function from_turing(
    chns=nothing;
    model=nothing,
    rng=Random.default_rng(),
    nchains=ndraws = chns isa Turing.MCMCChains.Chains ? last(size(chns)) : 1,
    ndraws=chns isa Turing.MCMCChains.Chains ? first(size(chns)) : 1_000,
    library=Turing,
    observed_data=nothing,
    constant_data=nothing,
    posterior_predictive=nothing,
    prior=nothing,
    prior_predictive=nothing,
    log_likelihood=nothing,
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
    model === nothing && return from_mcmcchains(chns; library=library, groups..., kwargs...)
    if groups[:prior] === nothing
        groups[:prior] = Turing.sample(rng, model, Turing.Prior(), Turing.MCMCThreads(), ndraws, nchains; progress=false)
    end

    groups[:observed_data] === nothing &&
        return from_mcmcchains(chns; library=library, groups..., kwargs...)

    observed_data = groups[:observed_data]
    data_var_names = Set(
        observed_data isa Dict ? Symbol.(keys(observed_data)) : propertynames(observed_data)
    )

    if groups[:constant_data] === nothing
        groups[:constant_data] = NamedTuple(
            filter(p -> first(p) ∉ data_var_names, pairs(model.args))
        )
    end

    # Instantiate the predictive model
model_predict = Turing.DynamicPPL.Model{data_var_names}(model.name, model.f, args_pred, model.defaults)

    # and then sample!
    if groups[:prior_predictive] === nothing && groups[:prior] isa Turing.MCMCChains.Chains
        groups[:prior_predictive] = Turing.predict(rng, model_predict, groups[:prior])
    end

    if chns isa Turing.MCMCChains.Chains
        if groups[:posterior_predictive] === nothing && chns isa Turing.MCMCChains.Chains
            groups[:posterior_predictive] = Turing.predict(rng, model_predict, chns)
        end

        if groups[:log_likelihood] === nothing &&
           groups[:posterior_predictive] isa MCMCChains.Chains
            loglikelihoods = Turing.pointwise_loglikelihoods(
                model, Turing.MCMCChains.get_sections(chns, :parameters)
            )

            # Bundle loglikelihoods into a `Chains` object so we can reuse our own variable
            # name parsing
            pred_names = string.(keys(groups[:posterior_predictive]))
            loglikelihoods_vals = getindex.(Ref(loglikelihoods), pred_names)
            loglikelihoods_arr = permutedims(cat(loglikelihoods_vals...; dims=3), (1, 3, 2))
            groups[:log_likelihood] = Turing.MCMCChains.Chains(
                loglikelihoods_arr, pred_names
            )
        end
    end

    return from_mcmcchains(chns; library=Turing, groups..., kwargs...)
end
