using ArviZ.InferenceObjects, DimensionalData, Test

@testset "conversion to Dataset" begin
    @testset "convert_to_dataset" begin
        nchains = 4
        ndraws = 100
        nshared = 3
        xdims = (:chain, :draw, :shared)
        x = DimArray(randn(nchains, ndraws, nshared), xdims)
        ydims = (:chain, :draw, Dim{:ydim1}(Any["a", "b"]), Dim{:shared})
        y = DimArray(randn(nchains, ndraws, 2, nshared), ydims)
        metadata = Dict(:prop1 => "val1", :prop2 => "val2")
        ds = ArviZ.Dataset((; x, y); metadata)

        @testset "ArviZ.convert_to_dataset(::ArviZ.Dataset; kwargs...)" begin
            @test ArviZ.convert_to_dataset(ds) isa ArviZ.Dataset
            @test ArviZ.convert_to_dataset(ds) === ds
        end

        @testset "ArviZ.convert_to_dataset(::$T; kwargs...)" for T in (Dict, NamedTuple)
            data = (x=randn(4, 100), y=randn(4, 100, 2))
            if T <: Dict
                data = T(pairs(data))
            end
            ds2 = ArviZ.convert_to_dataset(data)
            @test ds2 isa ArviZ.Dataset
            @test ds2.x == data[:x]
            @test DimensionalData.name(DimensionalData.dims(ds2.x)) == (:chain, :draw)
            @test ds2.y == data[:y]
            @test DimensionalData.name(DimensionalData.dims(ds2.y)) ==
                (:chain, :draw, :y_dim_1)
        end

        @testset "ArviZ.convert_to_dataset(::InferenceData; kwargs...)" begin
            idata = random_data()
            @test ArviZ.convert_to_dataset(idata) === idata.posterior
            @test ArviZ.convert_to_dataset(idata; group=:prior) === idata.prior
        end
    end
end
