@testset "InferenceData" begin
    data = load_arviz_data("centered_eight")

    @testset "construction" begin
        pydata = PyObject(data)
        @test InferenceData(pydata) isa InferenceData
        @test PyObject(InferenceData(pydata)) === pydata
        @test InferenceData(data) === data
        data2 = InferenceData(; posterior = data.posterior)
        @test data2 isa InferenceData
        @test :posterior in propertynames(data2)
    end

    @testset "properties" begin
        @test :posterior in propertynames(data)
        @test length(propertynames(data)) > 1
        data3 = InferenceData(; posterior = data.posterior, prior = data.prior)
        @test :prior in propertynames(data3)
        delete!(data3, :prior)
        @test :prior ∉ propertynames(data3)
    end

    @testset "conversion" begin
        @test pyisinstance(PyObject(data), ArviZ.arviz.InferenceData)
        data4 = convert(InferenceData, PyObject(data))
        @test data4 isa InferenceData
        @test PyObject(data4) === PyObject(data)
    end
end

@testset "+(::InferenceData, ::InferenceData)" begin
    rng = MersenneTwister(42)
    idata1 = from_dict(posterior = Dict(
        "A" => randn(rng, 2, 10, 2),
        "B" => randn(rng, 2, 10, 5, 2),
    ))
    idata2 = from_dict(prior = Dict(
        "C" => randn(rng, 2, 10, 2),
        "D" => randn(rng, 2, 10, 5, 2),
    ))

    new_idata = idata1 + idata2
    @test new_idata isa InferenceData
    @test check_multiple_attrs(
        Dict(:posterior => ["A", "B"], :prior => ["C", "D"]),
        new_idata,
    ) == []
end

@testset "concat" begin
    rng = MersenneTwister(42)
    idata1 = from_dict(posterior = Dict(
        "A" => randn(rng, 2, 10, 2),
        "B" => randn(rng, 2, 10, 5, 2),
    ))
    idata2 = from_dict(prior = Dict(
        "C" => randn(rng, 2, 10, 2),
        "D" => randn(rng, 2, 10, 5, 2),
    ))

    new_idata = concat(idata1, idata2)
    @test new_idata isa InferenceData
    @test check_multiple_attrs(
        Dict(:posterior => ["A", "B"], :prior => ["C", "D"]),
        new_idata,
    ) == []

    new_idata = concat!(idata1, idata2)
    @test new_idata === idata1
    @test check_multiple_attrs(
        Dict(:posterior => ["A", "B"], :prior => ["C", "D"]),
        idata1,
    ) == []
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
end

@testset "ArviZ.convert_to_dataset(data::InferenceData; kwargs...)" begin
    rng = MersenneTwister(42)
    A = Dict("A" => randn(rng, 2, 10, 2))
    B = Dict("B" => randn(rng, 2, 10, 2))
    dataA = ArviZ.convert_to_dataset(A)
    dataB = ArviZ.convert_to_dataset(B)
    idata = InferenceData(posterior = dataA, prior = dataB)

    ds1 = ArviZ.convert_to_dataset(idata)
    @test ds1 isa ArviZ.Dataset
    @test "A" ∈ [ds1.keys()...]

    ds2 = ArviZ.convert_to_dataset(idata; group = :prior)
    @test ds2 isa ArviZ.Dataset
    @test "B" ∈ [ds2.keys()...]
end

@testset "from_dict" begin
    rng = MersenneTwister(42)

    posterior = Dict("A" => randn(rng, 2, 10, 2), "B" => randn(rng, 2, 10, 5, 2))
    prior = Dict("C" => randn(rng, 2, 10, 2), "D" => randn(rng, 2, 10, 5, 2))

    idata = from_dict(posterior; prior = prior)
    @test check_multiple_attrs(
        Dict(:posterior => ["A", "B"], :prior => ["C", "D"]),
        idata,
    ) == []

    idata2 = from_dict(; prior = prior)
    @test check_multiple_attrs(Dict(:prior => ["C", "D"]), idata2) == []
end

@testset "netcdf roundtrip" begin
    data = load_arviz_data("centered_eight")
    mktemp() do path, _
        to_netcdf(data, path)
        data2 = from_netcdf(path)
        @test data2 isa InferenceData
        @test propertynames(data) == propertynames(data2)
    end
end

@testset "load_arviz_data" begin
    data = load_arviz_data("centered_eight")
    @test data isa InferenceData

    datasets = load_arviz_data()
    @test datasets isa Dict
end
