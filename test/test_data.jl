using MonteCarloMeasurements: Particles

@testset "InferenceData" begin
    data = load_arviz_data("centered_eight")

    @testset "construction" begin
        pydata = PyObject(data)
        @test InferenceData(pydata) isa InferenceData
        @test PyObject(InferenceData(pydata)) === pydata
        @test InferenceData(data) === data
        @test_throws ArgumentError InferenceData(py"PyNullObject()")
        data2 = InferenceData(; posterior=data.posterior)
        @test data2 isa InferenceData
        @test :posterior in propertynames(data2)
        @test hash(data) == hash(pydata)
    end

    @testset "properties" begin
        @test :posterior in propertynames(data)
        @test length(propertynames(data)) > 1
        data3 = InferenceData(; posterior=data.posterior, prior=data.prior)
        @test :prior in propertynames(data3)
        delete!(data3, :prior)
        @test :prior ∉ propertynames(data3)
    end

    @testset "groups" begin
        data4 = InferenceData(; posterior=data.posterior)
        @test ArviZ.groupnames(data4) == [:posterior]
        g = ArviZ.groups(data4)
        @test g isa Dict
        @test :posterior in keys(g)
    end

    @testset "isempty" begin
        @test !isempty(data)
        @test isempty(InferenceData())
    end

    @testset "conversion" begin
        @test pyisinstance(PyObject(data), ArviZ.arviz.InferenceData)
        data4 = convert(InferenceData, PyObject(data))
        @test data4 isa InferenceData
        @test PyObject(data4) === PyObject(data)

        # TODO: improve this test
        @test convert(InferenceData, [1.0, 2.0, 3.0, 4.0]) isa InferenceData
    end

    @testset "show" begin
        @testset "plain" begin
            text = sprint(show, data)
            @test startswith(text, "InferenceData with groups:")
            rest = split(PyObject(data).__str__(), '\n'; limit=2)[2]
            @test split(text, '\n'; limit=2)[2] == rest
        end

        @testset "html" begin
            text = repr(MIME("text/html"), data)
            @test text isa String
            @test !occursin("<xarray.Dataset>", text)
            @test !occursin("&lt;xarray.Dataset&gt;", text)
            @test !occursin("arviz.InferenceData", text)
            @test occursin("Dataset (xarray.Dataset)", text)
            @test occursin("InferenceData", text)
        end
    end
end

@testset "+(::InferenceData, ::InferenceData)" begin
    rng = MersenneTwister(42)
    idata1 = from_dict(;
        posterior=Dict("A" => randn(rng, 2, 10, 2), "B" => randn(rng, 2, 10, 5, 2))
    )
    idata2 = from_dict(;
        prior=Dict("C" => randn(rng, 2, 10, 2), "D" => randn(rng, 2, 10, 5, 2))
    )

    new_idata = idata1 + idata2
    @test new_idata isa InferenceData
    @test check_multiple_attrs(
        Dict(:posterior => ["A", "B"], :prior => ["C", "D"]), new_idata
    ) == []
end

@testset "concat" begin
    rng = MersenneTwister(42)
    idata1 = from_dict(;
        posterior=Dict("A" => randn(rng, 2, 10, 2), "B" => randn(rng, 2, 10, 5, 2))
    )
    idata2 = from_dict(;
        prior=Dict("C" => randn(rng, 2, 10, 2), "D" => randn(rng, 2, 10, 5, 2))
    )

    new_idata = concat(idata1, idata2)
    @test new_idata isa InferenceData
    @test check_multiple_attrs(
        Dict(:posterior => ["A", "B"], :prior => ["C", "D"]), new_idata
    ) == []

    new_idata = concat!(idata1, idata2)
    @test new_idata === idata1
    @test check_multiple_attrs(
        Dict(:posterior => ["A", "B"], :prior => ["C", "D"]), idata1
    ) == []

    idata3 = concat!()
    @test idata3 isa InferenceData
    @test isempty(idata3)

    idata4 = InferenceData()
    new_idata = concat!(idata4, idata1)
    @test new_idata === idata4
    @test check_multiple_attrs(
        Dict(:posterior => ["A", "B"], :prior => ["C", "D"]), idata4
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
