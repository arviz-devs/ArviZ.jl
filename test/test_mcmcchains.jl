using MCMCChains

function makechains(names, ndraws, nchains; seed = 42, internal_names = [])
    rng = MersenneTwister(seed)
    nvars = length(names)
    vals = randn(rng, ndraws, nvars, nchains)
    chns = Chains(vals, names, Dict(:internals => internal_names))
    return chns
end

function makechains(nvars::Int, args...; kwargs...)
    names = ["var$(i)" for i = 1:nvars]
    return makechains(names, args...; kwargs...)
end

dimsizes(ds) = ds._dims
convertindex(x::AbstractArray) = x
convertindex(o::PyObject) = o.array.values
vardict(ds) = Dict(k => convertindex(v._data) for (k, v) in ds._variables)
dimdict(ds) = Dict(k => v._dims for (k, v) in ds._variables)
attributes(ds) = ds.attrs

function test_chains_data(chns, idata, group, names; coords = Dict(), dims = Dict())
    ndraws, nvars, nchains = size(chns)
    @test idata isa InferenceData
    @test group in propertynames(idata)
    ds = getproperty(idata, group)
    sizes = dimsizes(ds)
    @test length(sizes) == 2 + length(coords)
    vars = vardict(ds)
    for name in names
        @test name in keys(vars)
        dim = get(dims, name, [])
        s = (x -> length(get(coords, x, []))).(dim)
        @test size(vars[name]) == (nchains, ndraws, s...)
    end
    @test attributes(ds)["inference_library"] == "MCMCChains"
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
        idata = from_mcmcchains(prior = chns)
        test_chains_data(chns, idata, :prior, names(chns))
    end

    @testset "posterior + prior" begin
        nvars, nchains, ndraws = 2, 4, 20
        chns = makechains(nvars, ndraws, nchains)
        idata = from_mcmcchains(chns; prior = chns)
        test_chains_data(chns, idata, :posterior, names(chns))
        test_chains_data(chns, idata, :prior, names(chns))
    end

    @testset "coords/dim" begin
        names = ["a[1]", "a[2]", "b[1]", "b[2]"]
        coords = Dict("ai" => 1:2, "bi" => ["b1", "b2"])
        dims = Dict("a" => ["ai"], "b" => ["bi"])
        nchains, ndraws = 4, 20
        chns = makechains(names, ndraws, nchains)
        idata = from_mcmcchains(chns; coords = coords, dims = dims)
        test_chains_data(chns, idata, :posterior, ["a", "b"]; coords = coords, dims = dims)
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
        idata = from_mcmcchains(chns; coords = coords, dims = dims)
        test_chains_data(chns, idata, :posterior, ["a"]; coords = coords, dims = dims)
        arr = vardict(idata.posterior)["a"]
        @test arr[:, :, 1, 1] == permutedims(chns.value[:, names[1], :], [2, 1])
        @test arr[:, :, 2, 2] == permutedims(chns.value[:, names[2], :], [2, 1])
        @test arr[:, :, 2, 1] == permutedims(chns.value[:, names[3], :], [2, 1])
        @test arr[:, :, 1, 2] == permutedims(chns.value[:, names[4], :], [2, 1])
    end

    @testset "complete" begin
        nchains, ndraws = 4, 20
        posterior = prior = ["a[1]", "a[2]"]
        posterior_predictive = prior_predictive = ["ahat[1]", "ahat[2]"]
        observed_data = ["xhat[1]", "xhat[2]"]
        constant_data = ["x[1]", "x[2]"]
        sample_stats = ["stat"]
        log_likelihood = "log_lik"
        post_names = [
            posterior
            posterior_predictive
            observed_data
            constant_data
            sample_stats
            log_likelihood
        ]
        prior_names = [prior; prior_predictive; sample_stats]
        chns = makechains(post_names, ndraws, nchains; internal_names = ["stat"])
        chns2 = makechains(prior_names, ndraws, nchains; internal_names = ["stat"])
        idata = from_mcmcchains(
            chns;
            posterior_predictive = "ahat",
            prior = chns2,
            prior_predictive = "ahat",
            observed_data = "xhat",
            constant_data = "x",
            log_likelihood = "log_lik",
        )
        for group in (
            :posterior,
            :prior,
            :posterior_predictive,
            :prior_predictive,
            :observed_data,
            :constant_data,
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
        @test length(dimdict(idata.sample_stats)) == 4
        @test "stat" ∈ keys(dimdict(idata.sample_stats))
        @test "log_likelihood" ∈ keys(dimdict(idata.sample_stats))
        @test length(dimdict(idata.sample_stats_prior)) == 3
        @test "stat" ∈ keys(dimdict(idata.sample_stats_prior))
        @test length(dimdict(idata.observed_data)) == 4
        @test "xhat" ∈ keys(dimdict(idata.observed_data))
        @test length(dimdict(idata.constant_data)) == 4
        @test "x" ∈ keys(dimdict(idata.constant_data))
    end

    @testset "missing -> NaN" begin
        rng = MersenneTwister(42)
        nvars, nchains, ndraws = 2, 4, 20
        vals = randn(rng, ndraws, nvars, nchains)
        vals = Array{Union{Float64,Missing},3}(vals)
        vals[1,1,1] = missing
        names = ["var$(i)" for i = 1:nvars]
        chns = Chains(vals, names)
        @test Missing <: eltype(chns.value)
        idata = from_mcmcchains(chns)
        vdict = vardict(idata.posterior)
        @test eltype(vdict["var1"]) <: Real
        @test isnan(vdict["var1"][1, 1, 1])
    end

    @testset "info -> attributes" begin
        nvars, nchains, ndraws = 2, 4, 20
        chns = makechains(nvars, ndraws, nchains)
        idata = from_mcmcchains(chns; library = "MyLib")
        attrs = idata.posterior.attrs
        @test attrs isa Dict
        @test attrs["inference_library"] == "MyLib"
        @test attrs["mcmcchains_summary"] isa Dict
        mcmcchains_summary = idata.posterior.attrs["mcmcchains_summary"]
        @test first(values(mcmcchains_summary)).__class__.__name__ == "DataFrame"
    end
end

@testset "convert_to_dataset(::Chains)" begin
    nvars, nchains, ndraws = 2, 4, 20
    chns = makechains(nvars, ndraws, nchains)
    ds = ArviZ.convert_to_dataset(chns)
    @test ds isa PyObject
    @test ds.__class__.__name__ == "Dataset"
end

@testset "convert_to_inference_data(::Chains)" begin
    nvars, nchains, ndraws = 2, 4, 20
    chns = makechains(nvars, ndraws, nchains)
    idata = convert_to_inference_data(chns; group = :posterior)
    @test idata isa InferenceData
    @test :posterior in propertynames(idata)
    idata = convert_to_inference_data(chns; group = :prior)
    @test idata isa InferenceData
    @test :prior in propertynames(idata)
end
