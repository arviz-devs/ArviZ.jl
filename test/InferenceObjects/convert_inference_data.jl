using ArviZ.InferenceObjects, DimensionalData, Test
using ArviZ.InferenceObjects: default_var_name, groupnames

@testset "conversion to InferenceData" begin
    @testset "default_var_name" begin
        x = randn(4, 5)
        @test default_var_name(x) === :x
        @test default_var_name(DimensionalData.DimArray(x, (:a, :b))) === :x
        @test default_var_name(DimensionalData.DimArray(x, (:a, :b); name=:y)) === :y
    end

    @testset "convert_to_inference_data" begin
        @testset "convert_to_inference_data(::AbstractDimStack)" begin
            ds = namedtuple_to_dataset((x=randn(4, 10), y=randn(4, 10, 5)))
            idata1 = convert_to_inference_data(ds; group=:prior)
            @test groupnames(idata1) == (:prior,)
            idata2 = InferenceData(; prior=ds)
            @test idata2 == idata1
            idata3 = convert_to_inference_data(parent(ds); group=:prior)
            @test idata3 == idata1
        end

        @testset "convert_to_inference_data(::$T)" for T in (NamedTuple, Dict)
            data = (A=randn(2, 10, 2), B=randn(2, 10, 5, 2))
            if T <: Dict
                data = Dict(pairs(data))
            end
            idata = convert_to_inference_data(data)
            check_idata_schema(idata)
            @test groupnames(idata) == (:posterior,)
            posterior = idata.posterior
            @test posterior.A == data[:A]
            @test posterior.B == data[:B]
            idata2 = convert_to_inference_data(data; group=:prior)
            check_idata_schema(idata2)
            @test groupnames(idata2) == (:prior,)
            @test idata2.prior == idata.posterior
        end

        @testset "convert_to_inference_data(::$T)" for T in (Array, DimensionalData.DimArray)
            data = randn(2, 10, 2)
            if T <: DimensionalData.DimArray
                data = DimensionalData.DimArray(data, (:a, :b, :c); name=:y)
            end
            idata = convert_to_inference_data(data)
            check_idata_schema(idata)
            @test groupnames(idata) == (:posterior,)
            posterior = idata.posterior
            if T <: DimensionalData.DimArray
                @test posterior.y == data
            else
                @test posterior.x == data
            end
            idata2 = convert_to_inference_data(data; group=:prior)
            check_idata_schema(idata2)
            @test groupnames(idata2) == (:prior,)
            @test idata2.prior == idata.posterior
        end
    end
end
