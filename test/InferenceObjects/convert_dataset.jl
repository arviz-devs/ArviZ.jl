using ArviZ.InferenceObjects, DimensionalData, Test

@testset "conversion to Dataset" begin
    @testset "conversion" begin
        J = 8
        K = 6
        L = 3
        nchains = 4
        ndraws = 500
        vars = (a=randn(nchains, ndraws, J), b=randn(nchains, ndraws, K, L))
        coords = (bi=2:(K + 1), draw=1:2:1_000)
        dims = (b=[:bi, nothing],)
        attrs = Dict(:mykey => 5)
        ds = namedtuple_to_dataset(vars; library="MyLib", coords, dims, attrs)
        @test convert(Dataset, ds) === ds
        ds2 = convert(Dataset, [1.0, 2.0, 3.0, 4.0])
        @test ds2 isa Dataset
        @test ds2 == convert_to_dataset([1.0, 2.0, 3.0, 4.0])
        @test convert(DimensionalData.DimStack, ds) === parent(ds)
    end

    @testset "convert_to_dataset" begin
        nchains = 4
        ndraws = 100
        nshared = 3
        xdims = (:chain, :draw, :shared)
        x = DimArray(randn(nchains, ndraws, nshared), xdims)
        ydims = (:chain, :draw, Dim{:ydim1}(Any["a", "b"]), Dim{:shared})
        y = DimArray(randn(nchains, ndraws, 2, nshared), ydims)
        metadata = Dict(:prop1 => "val1", :prop2 => "val2")
        ds = Dataset((; x, y); metadata)

        @testset "convert_to_dataset(::Dataset; kwargs...)" begin
            @test convert_to_dataset(ds) isa Dataset
            @test convert_to_dataset(ds) === ds
        end

        @testset "convert_to_dataset(::$T; kwargs...)" for T in (Dict, NamedTuple)
            data = (x=randn(4, 100), y=randn(4, 100, 2))
            if T <: Dict
                data = T(pairs(data))
            end
            ds2 = convert_to_dataset(data)
            @test ds2 isa Dataset
            @test ds2.x == data[:x]
            @test DimensionalData.name(DimensionalData.dims(ds2.x)) == (:chain, :draw)
            @test ds2.y == data[:y]
            @test DimensionalData.name(DimensionalData.dims(ds2.y)) ==
                (:chain, :draw, :y_dim_1)
        end

        @testset "convert_to_dataset(::InferenceData; kwargs...)" begin
            idata = random_data()
            @test convert_to_dataset(idata) === idata.posterior
            @test convert_to_dataset(idata; group=:prior) === idata.prior
        end
    end
end
