using Random
using PyCall

function create_model(seed = 10)
    rng = MersenneTwister(seed)
    J = 8
    nchains = 4
    ndraws = 500
    data = Dict(
        "J" => J,
        "y" => [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        "sigma" => [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
    )
    posterior = Dict(
        "mu" => randn(rng, nchains, ndraws),
        "tau" => abs.(randn(rng, nchains, ndraws)),
        "eta" => randn(rng, nchains, ndraws, J),
        "theta" => randn(rng, nchains, ndraws, J),
    )
    posterior_predictive = Dict("y" => randn(rng, nchains, ndraws, J))
    sample_stats = Dict(
        "energy" => randn(rng, nchains, ndraws),
        "diverging" => (randn(rng, nchains, ndraws) .> 0.90),
        "max_depth" => (randn(rng, nchains, ndraws) .> 0.90),
        "log_likelihood" => randn(rng, nchains, ndraws, J),
    )
    prior = Dict(
        "mu" => randn(rng, nchains, ndraws) / 2,
        "tau" => abs.(randn(rng, nchains, ndraws)) / 2,
        "eta" => randn(rng, nchains, ndraws, data["J"]) / 2,
        "theta" => randn(rng, nchains, ndraws, data["J"]) / 2,
    )
    prior_predictive = Dict("y" => randn(rng, nchains, ndraws, J) / 2)
    sample_stats_prior = Dict(
        "energy" => randn(rng, nchains, ndraws),
        "diverging" => Int.(randn(rng, nchains, ndraws) .> 0.95),
    )
    model = from_dict(
        posterior = posterior,
        posterior_predictive = posterior_predictive,
        sample_stats = sample_stats,
        prior = prior,
        prior_predictive = prior_predictive,
        sample_stats_prior = sample_stats_prior,
        observed_data = Dict("y" => data["y"]),
        dims = Dict("y" => ["obs_dim"], "log_likelihood" => ["obs_dim"]),
        coords = Dict("obs_dim" => 1:J),
    )
    return model
end

models() = (model_1 = create_model(10), model_2 = create_model(11))

function noncentered_schools_data()
    return Dict(
        "J" => 8,
        "y" => [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        "sigma" => [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
    )
end

function check_multiple_attrs(test_dict, parent)
    failed_attrs = []
    for (dataset_name, attributes) in test_dict
        if Symbol(dataset_name) ∈ propertynames(parent)
            dataset = getproperty(parent, Symbol(dataset_name))
            for attribute in attributes
                if Symbol(attribute) ∉ propertynames(dataset)
                    push!(failed_attrs, (dataset_name, attribute))
                end
            end
        else
            push!(failed_attrs, dataset_name)
        end
    end
    return failed_attrs
end
