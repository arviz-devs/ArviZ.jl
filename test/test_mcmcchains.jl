using MCMCChains: MCMCChains
using CmdStan, OrderedCollections

const noncentered_schools_stan_model = """
    data {
        int<lower=0> J;
        real y[J];
        real<lower=0> sigma[J];
    }
    parameters {
        real mu;
        real<lower=0> tau;
        real eta[J];
    }
    transformed parameters {
        real theta[J];
        for (j in 1:J)
            theta[j] = mu + tau * eta[j];
    }
    model {
        mu ~ normal(0, 5);
        tau ~ cauchy(0, 5);
        eta ~ normal(0, 1);
        y ~ normal(theta, sigma);
    }
    generated quantities {
        vector[J] log_lik;
        vector[J] y_hat;
        for (j in 1:J) {
            log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
            y_hat[j] = normal_rng(theta[j], sigma[j]);
        }
    }
"""

function makechains(
    names, ndraws, nchains, domains=[Float64 for _ in names]; seed=42, internal_names=[]
)
    rng = MersenneTwister(seed)
    nvars = length(names)
    var_vals = [rand(rng, domain, ndraws, nchains) for domain in domains]
    vals = permutedims(cat(var_vals...; dims=3), (1, 3, 2))
    chns = sort(MCMCChains.Chains(vals, names, Dict(:internals => internal_names)))
    return chns
end

function makechains(nvars::Int, args...; kwargs...)
    names = [Symbol("var$(i)") for i in 1:nvars]
    return makechains(names, args...; kwargs...)
end

function cmdstan_noncentered_schools(data, draws, chains; tmpdir=joinpath(pwd(), "tmp"))
    model_name = "school8"
    stan_model = Stanmodel(;
        name=model_name,
        model=noncentered_schools_stan_model,
        nchains=chains,
        num_warmup=draws,
        num_samples=draws,
        output_format=:mcmcchains,
        tmpdir,
    )
    rc, chns, cnames = stan(stan_model, data; summary=false)
    outfiles = ["$(tmpdir)/$(model_name)_samples_$(i).csv" for i in 1:chains]
    return (model=stan_model, files=outfiles, chains=chns)
end

function test_chains_data(chns, idata, group, names=names(chns); coords=(;), dims=(;))
    ndraws, nvars, nchains = size(chns)
    @test idata isa InferenceData
    @test group in ArviZ.groupnames(idata)
    ds = idata[group]
    for name in names
        @test name in keys(ds)
        dim = get(dims, name, ())
        s = (x -> length(get(coords, x, ()))).(dim)
        @test size(ds[name]) == (s..., ndraws, nchains)
    end
    @test ArviZ.attributes(ds)["inference_library"] == "MCMCChains"
    return nothing
end

@testset "from_mcmcchains" begin
    @testset "posterior" begin
        nvars, nchains, ndraws = 2, 4, 20
        chns = makechains(nvars, ndraws, nchains)
        idata = from_mcmcchains(chns)
        test_chains_data(chns, idata, :posterior, names(chns))
    end

    @testset "prior" begin
        nvars, nchains, ndraws = 2, 4, 20
        chns = makechains(nvars, ndraws, nchains)
        idata = from_mcmcchains(; prior=chns)
        test_chains_data(chns, idata, :prior, names(chns))
    end

    @testset "posterior + prior" begin
        nvars, nchains, ndraws = 2, 4, 20
        chns = makechains(nvars, ndraws, nchains)
        idata = from_mcmcchains(chns; prior=chns)
        test_chains_data(chns, idata, :posterior, names(chns))
        test_chains_data(chns, idata, :prior, names(chns))
    end

    @testset "coords/dim" begin
        var_names = Symbol.(["a[1]", "a[2]", "b[1]", "b[2]"])
        coords = (ai=1:2, bi=["b1", "b2"])
        dims = (a=(:ai,), b=[:bi])
        nchains, ndraws = 4, 20
        chns = makechains(var_names, ndraws, nchains)
        idata = from_mcmcchains(chns; coords, dims)
        test_chains_data(chns, idata, :posterior, [:a, :b]; coords, dims)
        var_dims = DimensionalData.layerdims(idata.posterior)
        @test Dimensions.name(var_dims[:a]) == (:ai, :draw, :chain)
        @test Dimensions.name(var_dims[:b]) == (:bi, :draw, :chain)
    end

    @testset "multivariate" begin
        var_names = Symbol.(["a[1][1]", "a.2.2", "a[2,1]", "a[1, 2]"])
        coords = (ai=1:2, aj=["aj1", "aj2"])
        dims = (a=[:ai, :aj],)
        nchains, ndraws = 4, 20
        chns = makechains(var_names, ndraws, nchains)

        # String or Symbol, which depends on MCMCChains version
        ET = Base.promote_typeof(chns.name_map.parameters...)

        idata = from_mcmcchains(chns; coords, dims)
        test_chains_data(chns, idata, :posterior, [:a]; coords, dims)
        arr = idata.posterior.a
        @test arr[1, 1, :, :] == chns.value[:, ET(var_names[1]), :]
        @test arr[2, 2, :, :] == chns.value[:, ET(var_names[2]), :]
        @test arr[2, 1, :, :] == chns.value[:, ET(var_names[3]), :]
        @test arr[1, 2, :, :] == chns.value[:, ET(var_names[4]), :]
    end

    @testset "specify eltypes" begin
        # https://github.com/arviz-devs/ArviZ.jl/issues/125
        nchains, ndraws = 4, 20
        var_names = [:x, :y, :z]
        domains = [Float64, (0, 1), 1:3]
        post = makechains(var_names, ndraws, nchains, domains)
        prior = makechains(var_names, ndraws, nchains, domains)
        posterior_predictive = makechains([:d], ndraws, nchains, [(0, 1)])
        idata = from_mcmcchains(
            post; prior, posterior_predictive, eltypes=(y=Bool, z=Int, d=Bool)
        )
        test_chains_data(post, idata, :posterior)
        test_chains_data(prior, idata, :prior)
        test_chains_data(posterior_predictive, idata, :posterior_predictive)
        @test eltype(idata.posterior[:y]) <: Bool
        @test eltype(idata.posterior[:z]) <: Int64
        @test eltype(idata.prior[:y]) <: Bool
        @test eltype(idata.prior[:z]) <: Int64
        @test eltype(idata.posterior_predictive[:d]) <: Bool

        idata2 = from_mcmcchains(
            post; prior, posterior_predictive=[:y], eltypes=(y=Bool, z=Int)
        )
        test_chains_data(post, idata2, :posterior, [:x, :z])
        test_chains_data(prior, idata2, :prior)
        @test eltype(idata2.posterior[:z]) <: Int64
        @test eltype(idata2.prior[:y]) <: Bool
        @test eltype(idata2.prior[:z]) <: Int64
        @test eltype(idata2.posterior_predictive[:y]) <: Bool
    end

    @testset "complete" begin
        nchains, ndraws = 4, 20
        nobs = 10
        posterior = prior = Symbol.(["a[1]", "a[2]"])
        posterior_predictive = prior_predictive = Symbol.(["ahat[1]", "ahat[2]"])
        observed_data = (xhat=1:nobs,)
        constant_data = (x=(1:nobs) .+ nobs,)
        predictions_constant_data = (z=(1:nobs) .+ nobs,)
        predictions = :zhat
        sample_stats = [:stat]
        log_likelihood = :log_lik
        post_names = [
            posterior
            posterior_predictive
            sample_stats
            predictions
            log_likelihood
        ]
        prior_names = [prior; prior_predictive; sample_stats]
        chns = makechains(post_names, ndraws, nchains; internal_names=[:stat])
        chns2 = makechains(prior_names, ndraws, nchains; internal_names=[:stat])
        idata = from_mcmcchains(
            chns;
            posterior_predictive=:ahat,
            predictions=:zhat,
            prior=chns2,
            prior_predictive=:ahat,
            observed_data,
            constant_data,
            predictions_constant_data,
            log_likelihood=:log_lik,
        )
        for group in (
            :posterior,
            :prior,
            :posterior_predictive,
            :prior_predictive,
            :predictions,
            :log_likelihood,
            :observed_data,
            :constant_data,
            :predictions_constant_data,
            :sample_stats,
            :sample_stats_prior,
        )
            @test group in propertynames(idata)
        end

        for group_name in (:prior, :posterior)
            @test length(Dimensions.dims(idata[group_name])) == 3
            @test :a ∈ keys(idata[group_name])
        end

        for group_name in (:prior_predictive, :posterior_predictive)
            @test length(Dimensions.dims(idata[group_name])) == 3
            @test :ahat ∈ keys(idata[group_name])
        end

        for group_name in (:sample_stats_prior, :sample_stats)
            @test length(Dimensions.dims(idata[group_name])) == 2
            @test :stat ∈ keys(idata[group_name])
        end

        @test length(Dimensions.dims(idata[:predictions])) == 2
        @test :zhat ∈ keys(idata[:predictions])

        @test length(Dimensions.dims(idata[:log_likelihood])) == 2
        @test :log_lik ∈ keys(idata[:log_likelihood])

        @test length(Dimensions.dims(idata[:observed_data])) == 1
        @test :xhat ∈ keys(idata[:observed_data])

        @test length(Dimensions.dims(idata[:constant_data])) == 1
        @test :x ∈ keys(idata[:constant_data])

        @test length(Dimensions.dims(idata[:predictions_constant_data])) == 1
        @test :z ∈ keys(idata[:predictions_constant_data])
    end

    # https://github.com/arviz-devs/ArviZ.jl/issues/146
    @testset "prior predictive w/o prior" begin
        nvars, nchains, ndraws = 2, 4, 20
        chns = makechains(nvars, ndraws, nchains)
        prior_predictive = randn(nchains, ndraws, 1)
        idata = from_mcmcchains(chns; prior_predictive)
        test_chains_data(chns, idata, :posterior, names(chns))
        @test :prior_predictive ∈ ArviZ.groupnames(idata)
        @test idata.prior_predictive.x ≈ prior_predictive

        prior_predictive = makechains(1, ndraws, nchains)
        idata = from_mcmcchains(chns; prior_predictive)
        test_chains_data(chns, idata, :posterior, names(chns))
        @test :prior_predictive ∈ ArviZ.groupnames(idata)
        @test idata.prior_predictive.var1 ≈
            permutedims(prior_predictive.value, (:var, :iter, :chain))[:var1, :, :]
    end

    @testset "missing -> NaN" begin
        rng = MersenneTwister(42)
        nvars, nchains, ndraws = 2, 4, 20
        vals = randn(rng, ndraws, nvars, nchains)
        vals = Array{Union{Float64,Missing},3}(vals)
        vals[1, 1, 1] = missing
        names = ["var$(i)" for i in 1:nvars]
        chns = MCMCChains.Chains(vals, names)
        @test Missing <: eltype(chns.value)
        idata = from_mcmcchains(chns)
        @test eltype(idata.posterior.var1) <: Real
        @test isnan(idata.posterior.var1[1, 1, 1])
    end

    @testset "info -> attributes" begin
        nvars, nchains, ndraws = 2, 4, 20
        chns = makechains(nvars, ndraws, nchains)
        idata = from_mcmcchains(chns; library="MyLib")
        metadata = DimensionalData.metadata(idata.posterior)
        @test metadata isa AbstractDict
        @test metadata["inference_library"] == "MyLib"
    end

    # https://github.com/arviz-devs/ArviZ.jl/issues/140
    @testset "large number of variables" begin
        num_vars = 1_000
        chn = MCMCChains.Chains(
            randn(100, num_vars, 1), [Symbol("x[$i]") for i in 1:num_vars]
        )
        @test hasproperty(from_mcmcchains(chn).posterior, :x)

        num_vars = 100
        chn = MCMCChains.Chains(
            randn(100, num_vars^2, 1),
            [Symbol("x[$i,$j]") for i in 1:num_vars for j in 1:num_vars],
        )
        @test hasproperty(from_mcmcchains(chn).posterior, :x)
    end
end

@testset "convert_to_dataset(::MCMCChains.Chains)" begin
    nvars, nchains, ndraws = 2, 4, 20
    chns = makechains(nvars, ndraws, nchains)
    ds = ArviZ.convert_to_dataset(chns; library="MyLib")
    @test ds isa ArviZ.Dataset
    metadata = DimensionalData.metadata(ds)
    @test metadata["inference_library"] == "MyLib"
end

@testset "convert_to_inference_data(::MCMCChains.Chains)" begin
    nvars, nchains, ndraws = 2, 4, 20
    chns = makechains(nvars, ndraws, nchains)
    idata = convert_to_inference_data(chns; group=:posterior)
    @test idata isa InferenceData
    @test :posterior in propertynames(idata)
    idata = convert_to_inference_data(chns; group=:prior)
    @test idata isa InferenceData
    @test :prior in propertynames(idata)
end

@testset "test MCMCChains readme example" begin
    # Define the experiment
    n_iter = 500
    n_name = 3
    n_chain = 2

    # experiment results
    val = randn(n_iter, n_name, n_chain) .+ [1, 2, 3]'
    val = hcat(val, rand(1:2, n_iter, 1, n_chain))

    # construct a Chains object
    chn = MCMCChains.Chains(val)  # According to version, this may introduce String or Symbol name

    @test ArviZ.summary(chn) !== nothing
end

@testset "from_cmdstan" begin
    data = noncentered_schools_data()
    mktempdir() do path
        output = cmdstan_noncentered_schools(data, 500, 4; tmpdir=path)
        posterior_predictive = prior_predictive = [:y_hat]
        log_likelihood = :log_lik
        coords = (school=1:8,)
        dims = (
            theta=[:school], y=[:school], log_lik=[:school], y_hat=[:school], eta=[:school]
        )
        idata1 = from_cmdstan(
            output.chains;
            posterior_predictive,
            log_likelihood,
            prior=output.chains,
            prior_predictive,
            coords,
            dims,
        )
        idata2 = from_cmdstan(
            output.files;
            posterior_predictive,
            log_likelihood,
            prior=output.files,
            prior_predictive,
            coords=Dict(pairs(coords)),
            dims=Dict(pairs(dims)),
        )
        @testset "idata.$(group)" for group in ArviZ.groupnames(idata2)
            ds1 = idata1[group]
            ds2 = idata2[group]

            for var_name in keys(ds1)
                da1 = ds1[var_name]
                da2 = ds2[var_name]
                @test Dimensions.name(Dimensions.dims(da1)) ==
                    reverse(Dimensions.name(Dimensions.dims(da2)))
                @test da1 ≈ permutedims(da2, reverse(Dimensions.dims(da2)))
            end
        end
    end
end
