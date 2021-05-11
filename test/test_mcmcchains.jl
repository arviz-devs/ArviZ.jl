using MCMCChains: MCMCChains
using CmdStan

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
    names = ["var$(i)" for i in 1:nvars]
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
        tmpdir=tmpdir,
    )
    rc, chns, cnames = stan(stan_model, data; summary=false)
    outfiles = ["$(tmpdir)/$(model_name)_samples_$(i).csv" for i in 1:chains]
    return (model=stan_model, files=outfiles, chains=chns)
end

function test_chains_data(chns, idata, group, names=names(chns); coords=Dict(), dims=Dict())
    ndraws, nvars, nchains = size(chns)
    @test idata isa InferenceData
    @test group in propertynames(idata)
    ds = getproperty(idata, group)
    sizes = dimsizes(ds)
    @test length(sizes) == 2 + length(coords)
    vars = vardict(ds)
    for name in names
        # `vars`, the value in ArviZ/Python is always String, 
        # while `names` is String or Symbol which depends on version of MCMCChains
        name = string(name)

        @test name in keys(vars)
        dim = get(dims, name, [])
        s = (x -> length(get(coords, x, []))).(dim)
        @test size(vars[name]) == (nchains, ndraws, s...)
    end
    @test attributes(ds)["inference_library"] == "MCMCChains"
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
        names = ["a[1]", "a[2]", "b[1]", "b[2]"]
        coords = Dict("ai" => 1:2, "bi" => ["b1", "b2"])
        dims = Dict("a" => ["ai"], "b" => ["bi"])
        nchains, ndraws = 4, 20
        chns = makechains(names, ndraws, nchains)
        idata = from_mcmcchains(chns; coords=coords, dims=dims)
        test_chains_data(chns, idata, :posterior, ["a", "b"]; coords=coords, dims=dims)
        vardims = dimdict(idata.posterior)
        @test vardims["a"] == ("chain", "draw", "ai")
        @test vardims["b"] == ("chain", "draw", "bi")
    end

    @testset "multivariate" begin
        names = ["a[1][1]", "a.2.2", "a[2,1]", "a[1, 2]"]
        coords = Dict("ai" => 1:2, "aj" => ["aj1", "aj2"])
        dims = Dict("a" => ["ai", "aj"])
        nchains, ndraws = 4, 20
        chns = makechains(names, ndraws, nchains)

        # String or Symbol, which depends on MCMCChains version
        ET = Base.promote_typeof(chns.name_map.parameters...)

        idata = from_mcmcchains(chns; coords=coords, dims=dims)
        test_chains_data(chns, idata, :posterior, ["a"]; coords=coords, dims=dims)
        arr = vardict(idata.posterior)["a"]
        @test arr[:, :, 1, 1] == permutedims(chns.value[:, ET(names[1]), :], [2, 1])
        @test arr[:, :, 2, 2] == permutedims(chns.value[:, ET(names[2]), :], [2, 1])
        @test arr[:, :, 2, 1] == permutedims(chns.value[:, ET(names[3]), :], [2, 1])
        @test arr[:, :, 1, 2] == permutedims(chns.value[:, ET(names[4]), :], [2, 1])
    end

    @testset "specify eltypes" begin
        # https://github.com/arviz-devs/ArviZ.jl/issues/125
        nchains, ndraws = 4, 20
        names = ["x", "y", "z"]
        domains = [Float64, (0, 1), 1:3]
        post = makechains(names, ndraws, nchains, domains)
        prior = makechains(names, ndraws, nchains, domains)
        post_pred = makechains(["d"], ndraws, nchains, [(0, 1)])
        idata = from_mcmcchains(
            post;
            prior=prior,
            posterior_predictive=post_pred,
            eltypes=Dict("y" => Bool, "z" => Int, "d" => Bool),
        )
        test_chains_data(post, idata, :posterior)
        test_chains_data(prior, idata, :prior)
        test_chains_data(post_pred, idata, :posterior_predictive)
        @test eltype(idata.posterior[:y].values) <: Bool
        @test eltype(idata.posterior[:z].values) <: Int64
        @test eltype(idata.prior[:y].values) <: Bool
        @test eltype(idata.prior[:z].values) <: Int64
        @test eltype(idata.posterior_predictive[:d].values) <: Bool

        idata2 = from_mcmcchains(
            post;
            prior=prior,
            posterior_predictive=["y"],
            eltypes=Dict("y" => Bool, "z" => Int),
        )
        test_chains_data(post, idata2, :posterior, ["x", "z"])
        test_chains_data(prior, idata2, :prior)
        @test eltype(idata2.posterior[:z].values) <: Int64
        @test eltype(idata2.prior[:y].values) <: Bool
        @test eltype(idata2.prior[:z].values) <: Int64
        @test eltype(idata2.posterior_predictive[:y].values) <: Bool
    end

    @testset "complete" begin
        nchains, ndraws = 4, 20
        nobs = 10
        posterior = prior = ["a[1]", "a[2]"]
        posterior_predictive = prior_predictive = ["ahat[1]", "ahat[2]"]
        observed_data = Dict("xhat" => 1:nobs)
        constant_data = Dict("x" => (1:nobs) .+ nobs)
        predictions_constant_data = Dict("z" => (1:nobs) .+ nobs)
        predictions = "zhat"
        sample_stats = ["stat"]
        log_likelihood = "log_lik"
        post_names = [
            posterior
            posterior_predictive
            sample_stats
            predictions
            log_likelihood
        ]
        prior_names = [prior; prior_predictive; sample_stats]
        chns = makechains(post_names, ndraws, nchains; internal_names=["stat"])
        chns2 = makechains(prior_names, ndraws, nchains; internal_names=["stat"])
        idata = from_mcmcchains(
            chns;
            posterior_predictive="ahat",
            predictions="zhat",
            prior=chns2,
            prior_predictive="ahat",
            observed_data=observed_data,
            constant_data=constant_data,
            predictions_constant_data=predictions_constant_data,
            log_likelihood="log_lik",
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
        @test length(dimdict(idata.posterior)) == 4
        @test "a" ∈ keys(dimdict(idata.posterior))
        @test length(dimdict(idata.posterior_predictive)) == 4
        @test "ahat" ∈ keys(dimdict(idata.posterior_predictive))
        @test length(dimdict(idata.prior)) == 4
        @test "a" ∈ keys(dimdict(idata.prior))
        @test length(dimdict(idata.prior_predictive)) == 4
        @test "ahat" ∈ keys(dimdict(idata.prior_predictive))
        @test length(dimdict(idata.sample_stats)) == 3
        @test "stat" ∈ keys(dimdict(idata.sample_stats))
        @test length(dimdict(idata.predictions)) == 3
        @test "zhat" ∈ keys(dimdict(idata.predictions))
        @test length(dimdict(idata.log_likelihood)) == 3
        @test "log_lik" ∈ keys(dimdict(idata.log_likelihood))
        @test length(dimdict(idata.sample_stats_prior)) == 3
        @test "stat" ∈ keys(dimdict(idata.sample_stats_prior))
        @test length(dimdict(idata.observed_data)) == 2
        @test "xhat" ∈ keys(dimdict(idata.observed_data))
        @test length(dimdict(idata.constant_data)) == 2
        @test "x" ∈ keys(dimdict(idata.constant_data))
        @test length(dimdict(idata.predictions_constant_data)) == 2
        @test "z" ∈ keys(dimdict(idata.predictions_constant_data))
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
        vdict = vardict(idata.posterior)
        @test eltype(vdict["var1"]) <: Real
        @test isnan(vdict["var1"][1, 1, 1])
    end

    @testset "info -> attributes" begin
        nvars, nchains, ndraws = 2, 4, 20
        chns = makechains(nvars, ndraws, nchains)
        idata = from_mcmcchains(chns; library="MyLib")
        attrs = idata.posterior.attrs
        @test attrs isa Dict
        @test attrs["inference_library"] == "MyLib"
    end
end

@testset "convert_to_dataset(::MCMCChains.Chains)" begin
    nvars, nchains, ndraws = 2, 4, 20
    chns = makechains(nvars, ndraws, nchains)
    ds = ArviZ.convert_to_dataset(chns; library="MyLib")
    @test ds isa ArviZ.Dataset
    attrs = ds.attrs
    @test "inference_library" ∈ keys(attrs)
    @test attrs["inference_library"] == "MyLib"
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

VERSION > v"1.0" && @testset "from_cmdstan" begin
    data = noncentered_schools_data()
    mktempdir() do path
        output = cmdstan_noncentered_schools(data, 500, 4; tmpdir=path)
        posterior_predictive = prior_predictive = ["y_hat"]
        log_likelihood = "log_lik"
        coords = Dict("school" => 1:8)
        dims = Dict(
            "theta" => ["school"],
            "y" => ["school"],
            "log_lik" => ["school"],
            "y_hat" => ["school"],
            "eta" => ["school"],
        )
        idata1 = from_cmdstan(
            output.chains;
            posterior_predictive=posterior_predictive,
            log_likelihood=log_likelihood,
            prior=output.chains,
            prior_predictive=prior_predictive,
            coords=coords,
            dims=dims,
        )
        idata2 = from_cmdstan(
            output.files;
            posterior_predictive=posterior_predictive,
            log_likelihood=log_likelihood,
            prior=output.files,
            prior_predictive=prior_predictive,
            coords=coords,
            dims=dims,
        )
        @testset "idata.$(group)" for group in Symbol.(idata2._groups)
            ds1 = getproperty(idata1, group)
            ds2 = getproperty(idata2, group)
            for f in (vardict, dimsizes)
                d1 = f(ds1)
                d2 = f(ds2)
                for k in keys(d1)
                    @test k in keys(d2)
                    @test d1[k] ≈ d2[k]
                end
            end
        end
    end
end
