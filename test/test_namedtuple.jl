using MonteCarloMeasurements: Particles

function test_namedtuple_data(
    idata, group, names, nchains, ndraws; library="MyLib", coords=Dict(), dims=Dict()
)
    @test idata isa InferenceData
    @test group in ArviZ.groupnames(idata)
    ds = getproperty(idata, group)
    sizes = dimsizes(ds)
    @test length(sizes) == 2 + length(coords)
    vars = vardict(ds)
    for name in string.(names)
        @test name in keys(vars)
        dim = get(dims, name, get(dims, Symbol(name), []))
        s = (x -> length(get(coords, x, get(coords, Symbol(x), [])))).(dim)
        @test size(vars[name]) == (nchains, ndraws, s...)
    end
    @test "inference_library" in keys(attributes(ds))
    @test attributes(ds)["inference_library"] == library
    return nothing
end

@testset "from_namedtuple" begin
    rng = MersenneTwister(42)

    nchains, ndraws = 4, 10
    sizes = (x=(), y=(2,), z=(3, 4))
    dims = Dict("y" => ["yx"], "z" => ["zx", "zy"])
    coords = Dict("yx" => ["y1", "y2"], "zx" => 1:3, "zy" => 1:4)

    nts = [
        "NamedTuple" =>
            (; (k => randn(rng, nchains, ndraws, v...) for (k, v) in pairs(sizes))...),
        "Vector{NamedTuple}" => [
            (; (k => randn(rng, ndraws, v...) for (k, v) in pairs(sizes))...) for
            _ in 1:nchains
        ],
        "Matrix{NamedTuple}" => [
            (; (k => randn(rng, v...) for (k, v) in pairs(sizes))...) for
            _ in 1:nchains, _ in 1:ndraws
        ],
        "Vector{Vector{NamedTuple}}" => [
            [(; (k => randn(rng, v...) for (k, v) in pairs(sizes))...) for _ in 1:ndraws] for _ in 1:nchains
        ],
        "Vector{NamedTuple} particles" => [
            (; (k => Particles(randn(rng, ndraws, v...)) for (k, v) in pairs(sizes))...) for _ in 1:nchains
        ],
    ]

    @testset "posterior::$(type)" for (type, nt) in nts
        idata1 = from_namedtuple(nt; dims=dims, coords=coords, library="MyLib")
        test_namedtuple_data(
            idata1,
            :posterior,
            keys(sizes),
            nchains,
            ndraws;
            library="MyLib",
            coords=coords,
            dims=dims,
        )

        idata2 = convert_to_inference_data(nt; dims=dims, coords=coords, library="MyLib")
        test_namedtuple_data(
            idata2,
            :posterior,
            keys(sizes),
            nchains,
            ndraws;
            library="MyLib",
            coords=coords,
            dims=dims,
        )
    end

    @testset "$(group)" for group in [
        :posterior_predictive, :sample_stats, :predictions, :log_likelihood
    ]
        @testset "::$(type)" for (type, nt) in nts
            idata1 = convert_to_inference_data(
                nt; (group => nt,)..., dims=dims, coords=coords, library="MyLib"
            )
            test_namedtuple_data(
                idata1,
                :posterior,
                keys(sizes),
                nchains,
                ndraws;
                library="MyLib",
                coords=coords,
                dims=dims,
            )
            test_namedtuple_data(
                idata1,
                group,
                keys(sizes),
                nchains,
                ndraws;
                library="MyLib",
                coords=coords,
                dims=dims,
            )

            idata2 = convert_to_inference_data(
                nt; (group => keys(sizes),)..., dims=dims, coords=coords, library="MyLib"
            )
            test_namedtuple_data(
                idata2,
                group,
                keys(sizes),
                nchains,
                ndraws;
                library="MyLib",
                coords=coords,
                dims=dims,
            )
        end
    end

    @testset "$(group)" for group in [:prior_predictive, :sample_stats_prior]
        @testset "::$(type)" for (type, nt) in nts
            idata1 = from_namedtuple(;
                prior=nt, (group => nt,)..., dims=dims, coords=coords, library="MyLib"
            )
            test_namedtuple_data(
                idata1,
                :prior,
                keys(sizes),
                nchains,
                ndraws;
                library="MyLib",
                coords=coords,
                dims=dims,
            )
            test_namedtuple_data(
                idata1,
                group,
                keys(sizes),
                nchains,
                ndraws;
                library="MyLib",
                coords=coords,
                dims=dims,
            )

            idata2 = from_namedtuple(;
                prior=nt,
                (group => keys(sizes),)...,
                dims=dims,
                coords=coords,
                library="MyLib",
            )
            test_namedtuple_data(
                idata2,
                group,
                keys(sizes),
                nchains,
                ndraws;
                library="MyLib",
                coords=coords,
                dims=dims,
            )
        end
    end

    @testset "$(group)" for group in
                            [:observed_data, :constant_data, :predictions_constant_data]
        _, nt = nts[1]
        idata = from_namedtuple(
            nt;
            (group => Dict("w" => [1.0, 2.0]),)...,
            dims=Dict("w" => ["wx"]),
            coords=Dict("wx" => 1:2),
            library="MyLib",
        )
        @test idata isa InferenceData
        @test group in ArviZ.groupnames(idata)
        ds = getproperty(idata, group)
        sizes = dimsizes(ds)
        @test length(sizes) == 1
        @test "w" in keys(vardict(ds))
        @test dimdict(ds)["w"] == ("wx",)
        @test "inference_library" in keys(attributes(ds))
        @test attributes(ds)["inference_library"] == "MyLib"

        # ensure that dims are matched to named tuple keys
        # https://github.com/arviz-devs/ArviZ.jl/issues/96
        idata = from_namedtuple(
            nt;
            (group => (w=[1.0, 2.0],),)...,
            dims=Dict("w" => ["wx"]),
            coords=Dict("wx" => 1:2),
            library="MyLib",
        )
        @test idata isa InferenceData
        @test group in ArviZ.groupnames(idata)
        ds = getproperty(idata, group)
        sizes = dimsizes(ds)
        @test length(sizes) == 1
        @test "w" in keys(vardict(ds))
        @test dimdict(ds)["w"] == ("wx",)
        @test "inference_library" in keys(attributes(ds))
        @test attributes(ds)["inference_library"] == "MyLib"
    end
end
