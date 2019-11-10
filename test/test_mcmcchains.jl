using MCMCChains

function makechains(names, ndraws, nchains; seed = 42)
    rng = MersenneTwister(seed)
    nvars = length(names)
    vals = randn(rng, ndraws, nvars, nchains)
    chns = Chains(vals, names)
    return chns
end

function makechains(nvars::Int, args...; kwargs...)
    names = ["var$(i)" for i in 1:nvars]
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
end
