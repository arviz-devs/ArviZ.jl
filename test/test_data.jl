using ArviZ, DimensionalData, Test
using MonteCarloMeasurements: Particles

@testset "InferenceData" begin
    var_names = (:a, :b)
    data_names = (:y,)
    coords = (
        chain=1:4, draw=1:100, shared=["s1", "s2", "s3"], dima=1:4, dimb=2:6, dimy=1:5
    )
    dims = (a=(:shared, :dima), b=(:shared, :dimb), y=(:shared, :dimy))
    metadata = (inference_library="PPL",)
    posterior = random_dataset(var_names, dims, coords, metadata)
    prior = random_dataset(var_names, dims, coords, metadata)
    group_data = (; prior, posterior)

    @testset "constructors" begin
        idata = @inferred(InferenceData(group_data))
        @test idata isa InferenceData
        @test getfield(idata, :groups) === (; posterior, prior)

        @test InferenceData(; group_data...) == idata
        @test InferenceData(idata) === idata
    end

    idata = InferenceData(group_data)

    @testset "properties" begin
        @test propertynames(idata) === (:posterior, :prior)
        @test getproperty(idata, :posterior) === posterior
        @test getproperty(idata, :prior) === prior
        @test hasproperty(idata, :posterior)
        @test hasproperty(idata, :prior)
        @test !hasproperty(idata, :prior_predictive)
    end

    @testset "iteration" begin
        @test keys(idata) === (:posterior, :prior)
        @test haskey(idata, :posterior)
        @test haskey(idata, :prior)
        @test !haskey(idata, :log_likelihood)
        @test values(idata) === (posterior, prior)
        @test pairs(idata) isa Base.Pairs
        @test Tuple(pairs(idata)) === (:posterior => posterior, :prior => prior)
        @test length(idata) == 2
        @test iterate(idata) === (posterior, 2)
        @test iterate(idata, 2) === (prior, 3)
        @test iterate(idata, 3) === nothing
        @test eltype(idata) <: ArviZ.Dataset
        @test collect(idata) isa Vector{<:ArviZ.Dataset}
    end

    @testset "indexing" begin
        @test idata[:posterior] === posterior
        @test idata[:prior] === prior
        @test idata[1] === posterior
        @test idata[2] === prior
        idata2 = Base.setindex(idata, posterior, :warmup_posterior)
        @test keys(idata2) === (:posterior, :prior, :warmup_posterior)
        @test idata2[:warmup_posterior] === posterior
    end

    @testset "isempty" begin
        @test !isempty(idata)
        @test isempty(InferenceData())
    end

    @testset "groups" begin
        @test ArviZ.groups(idata) === (; posterior, prior)
        @test ArviZ.groups(InferenceData(; prior)) === (; prior)
    end

    @testset "hasgroup" begin
        @test ArviZ.hasgroup(idata, :posterior)
        @test ArviZ.hasgroup(idata, :prior)
        @test !ArviZ.hasgroup(idata, :prior_predictive)
    end

    @testset "groupnames" begin
        @test ArviZ.groupnames(idata) === (:posterior, :prior)
        @test ArviZ.groupnames(InferenceData(; posterior)) === (:posterior,)
    end

    @testset "conversion" begin
        @test convert(InferenceData, idata) === idata
        @test convert(NamedTuple, idata) === parent(idata)
        @test NamedTuple(idata) === parent(idata)
    end

    @testset "show" begin
        @testset "plain" begin
            text = sprint(show, MIME("text/plain"), idata)
            @test text == """
            InferenceData with groups:
                > posterior
                > prior"""
        end

        @testset "html" begin
            # TODO: improve test
            text = sprint(show, MIME("text/html"), idata)
            @test text isa String
            @test occursin("InferenceData", text)
            @test occursin("Dataset", text)
        end
    end
end

@testset "extract_dataset" begin
    idata = random_data()
    post = extract_dataset(idata, :posterior; combined=false)
    for k in keys(idata.posterior)
        @test haskey(post, k)
        @test post[k] == idata.posterior[k]
        dims = DimensionalData.dims(post)
        dims_exp = DimensionalData.dims(idata.posterior)
        @test DimensionalData.name(dims) === DimensionalData.name(dims_exp)
        @test DimensionalData.index(dims) == DimensionalData.index(dims_exp)
    end
    prior = extract_dataset(idata, :prior; combined=false)
    for k in keys(idata.prior)
        @test haskey(prior, k)
        @test prior[k] == idata.prior[k]
        dims = DimensionalData.dims(prior)
        dims_exp = DimensionalData.dims(idata.prior)
        @test DimensionalData.name(dims) === DimensionalData.name(dims_exp)
        @test DimensionalData.index(dims) == DimensionalData.index(dims_exp)
    end
end

@testset "concat" begin
    data = random_data()
    idata1 = InferenceData(; posterior=data.posterior)
    idata2 = InferenceData(; prior=data.prior)
    new_idata = concat(idata1, idata2)
    @test new_idata isa InferenceData
    @test ArviZ.groups(new_idata) == (; posterior=data.posterior, prior=data.prior)
end

@testset "ArviZ.convert_to_dataset(::InferenceData; kwargs...)" begin
    A = Dict("A" => randn(rng, 2, 10, 2))
    B = Dict("B" => randn(rng, 2, 10, 2))
    dataA = ArviZ.convert_to_dataset(A)
    dataB = ArviZ.convert_to_dataset(B)
    idata = InferenceData(; posterior=dataA, prior=dataB)

    ds1 = ArviZ.convert_to_dataset(idata)
    @test ds1 isa ArviZ.Dataset
    @test "A" ∈ [ds1.keys()...]

    ds2 = ArviZ.convert_to_dataset(idata; group=:prior)
    @test ds2 isa ArviZ.Dataset
    @test "B" ∈ [ds2.keys()...]
end

@testset "convert_to_inference_data" begin
    rng = MersenneTwister(42)

    @testset "convert_to_inference_data(::Dict)" begin
        dataset = Dict("A" => randn(rng, 2, 10, 2), "B" => randn(rng, 2, 10, 5, 2))
        idata1 = convert_to_inference_data(dataset)
        @test idata1 isa InferenceData
        @test check_multiple_attrs(Dict(:posterior => ["A", "B"]), idata1) == []
    end

    @testset "convert_to_inference_data(::Array)" begin
        arr = randn(rng, 2, 10, 2)
        idata2 = convert_to_inference_data(arr)
        @test check_multiple_attrs(Dict(:posterior => ["x"]), idata2) == []
    end

    @testset "convert_to_inference_data(::Nothing)" begin
        idata3 = convert_to_inference_data(nothing)
        @test idata3 isa InferenceData
        @test isempty(idata3)
    end

    @testset "convert_to_inference_data(::Particles)" begin
        p = Particles(randn(rng, 10))
        idata4 = convert_to_inference_data(p)
        @test check_multiple_attrs(Dict(:posterior => ["x"]), idata4) == []
    end

    @testset "convert_to_inference_data(::Vector{Particles})" begin
        p = [Particles(randn(rng, 10)) for _ in 1:4]
        idata5 = convert_to_inference_data(p)
        @test check_multiple_attrs(Dict(:posterior => ["x"]), idata5) == []
    end
    @testset "convert_to_inference_data(::Vector{Array{Particles}})" begin
        p = [Particles(randn(rng, 10, 3)) for _ in 1:4]
        idata6 = convert_to_inference_data(p)
        @test check_multiple_attrs(Dict(:posterior => ["x"]), idata6) == []
    end
end

@testset "from_dict" begin
    rng = MersenneTwister(42)

    posterior = Dict("A" => randn(rng, 2, 10, 2), "B" => randn(rng, 2, 10, 5, 2))
    prior = Dict("C" => randn(rng, 2, 10, 2), "D" => randn(rng, 2, 10, 5, 2))

    idata = from_dict(posterior; prior)
    @test check_multiple_attrs(
        Dict(:posterior => ["A", "B"], :prior => ["C", "D"]), idata
    ) == []

    idata2 = from_dict(; prior)
    @test check_multiple_attrs(Dict(:prior => ["C", "D"]), idata2) == []
end

@testset "netcdf roundtrip" begin
    data = load_arviz_data("centered_eight")
    mktempdir() do path
        filename = joinpath(path, "tmp.nc")
        to_netcdf(data, filename)
        data2 = from_netcdf(filename)
        @test data2 isa InferenceData
        @test propertynames(data) == propertynames(data2)
        return nothing
    end
end

@testset "load_arviz_data" begin
    data = load_arviz_data("centered_eight")
    @test data isa InferenceData

    datasets = load_arviz_data()
    @test datasets isa Dict
end
