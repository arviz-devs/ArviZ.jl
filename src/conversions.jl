@doc doc"""
    from_mcmcchains(posterior::MCMCChains.Chains; kwargs...) -> InferenceData
    from_mcmcchains(; kwargs...) -> InferenceData
    from_mcmcchains(
        posterior::MCMCChains.Chains,
        posterior_predictive,
        predictions,
        log_likelihood;
        kwargs...
    ) -> InferenceData

Convert data in an `MCMCChains.Chains` format into an [`InferenceData`](@ref).

Any keyword argument below without an an explicitly annotated type above is allowed, so long
as it can be passed to [`convert_to_inference_data`](@ref).

# Arguments

- `posterior::MCMCChains.Chains`: Draws from the posterior

# Keywords

- `posterior_predictive::Any=nothing`: Draws from the posterior predictive distribution or
    name(s) of predictive variables in `posterior`
- `predictions`: Out-of-sample predictions for the posterior.
- `prior`: Draws from the prior
- `prior_predictive`: Draws from the prior predictive distribution or name(s) of predictive
    variables in `prior`
- `observed_data`: Observed data on which the `posterior` is conditional. It should only
    contain data which is modeled as a random variable. Keys are parameter names and values.
- `constant_data`: Model constants, data included in the model that are not modeled as
    random variables. Keys are parameter names.
- `predictions_constant_data`: Constants relevant to the model predictions (i.e. new `x`
    values in a linear regression).
- `log_likelihood`: Pointwise log-likelihood for the data. It is recommended to use this
    argument as a named tuple whose keys are observed variable names and whose values are log
    likelihood arrays. Alternatively, provide the name of variable in `posterior` containing
    log likelihoods.
- `library=MCMCChains`: Name of library that generated the chains
- `coords`: Map from named dimension to named indices
- `dims`: Map from variable name to names of its dimensions
- `eltypes`: Map from variable names to eltypes. This is primarily used to assign discrete
    eltypes to discrete variables that were stored in `Chains` as floats.

# Returns

- `InferenceData`: The data with groups corresponding to the provided data
"""
function from_mcmcchains end
