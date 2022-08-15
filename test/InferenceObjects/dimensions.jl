using ArviZ.InferenceObjects, DimensionalData, OffsetArrays, Test

@testset "dimension-related functions" begin
    @testset "has_all_sample_dims" begin
        @test !InferenceObjects.has_all_sample_dims((:chain,))
        @test !InferenceObjects.has_all_sample_dims((:draw,))
        @test InferenceObjects.has_all_sample_dims((:chain, :draw))
        @test InferenceObjects.has_all_sample_dims((:draw, :chain))
        @test InferenceObjects.has_all_sample_dims((:draw, :chain, :x))

        @test !InferenceObjects.has_all_sample_dims((Dim{:chain},))
        @test !InferenceObjects.has_all_sample_dims((Dim{:draw},))
        @test InferenceObjects.has_all_sample_dims((Dim{:chain}, Dim{:draw}))
        @test InferenceObjects.has_all_sample_dims((Dim{:draw}, Dim{:chain}))
        @test InferenceObjects.has_all_sample_dims((Dim{:draw}, Dim{:chain}, Dim{:x}))

        @test !InferenceObjects.has_all_sample_dims((Dim{:chain}(1:4),))
        @test !InferenceObjects.has_all_sample_dims((Dim{:draw}(1:10),))
        @test InferenceObjects.has_all_sample_dims((Dim{:chain}(1:4), Dim{:draw}(1:10)))
        @test InferenceObjects.has_all_sample_dims((Dim{:draw}(1:10), Dim{:chain}(1:4)))
        @test InferenceObjects.has_all_sample_dims((
            Dim{:draw}(1:10), Dim{:chain}(1:4), Dim{:x}(1:2)
        ))
    end

    @testset "as_dimension" begin
        coords = (;)
        @testset for dim in (:foo, Dim{:foo}, Dim{:foo,Colon})
            @test InferenceObjects.as_dimension(dim, coords, 2:10) === Dim{:foo}(2:10)
            dim === :foo || @inferred InferenceObjects.as_dimension(dim, coords, 2:10)
        end
        @test InferenceObjects.as_dimension(Dim{:foo}(1:5), coords, 2:10) === Dim{:foo}(1:5)
        coords = (; foo=3:8)
        @testset for dim in (:foo, Dim{:foo}, Dim{:foo,Colon}, Dim{:foo}(1:5))
            @test InferenceObjects.as_dimension(dim, coords, 2:10) === Dim{:foo}(3:8)
            dim === :foo || @inferred InferenceObjects.as_dimension(dim, coords, 2:10)
        end
    end

    @testset "generate_dims" begin
        x = OffsetArray(randn(4, 10, 2, 3), 0:3, 11:20, -1:0, 2:4)
        gdims = @inferred NTuple{4,Dimensions.Dimension} InferenceObjects.generate_dims(
            x, :x
        )
        @test gdims isa NTuple{4,Dim}
        @test Dimensions.name(gdims) === (:x_dim_1, :x_dim_2, :x_dim_3, :x_dim_4)
        @test Dimensions.index(gdims) == (0:3, 11:20, -1:0, 2:4)

        gdims = @inferred NTuple{4,Dimensions.Dimension} InferenceObjects.generate_dims(
            x, :y; dims=(:a, :b)
        )
        @test gdims isa NTuple{4,Dim}
        @test Dimensions.name(gdims) === (:a, :b, :y_dim_3, :y_dim_4)
        @test Dimensions.index(gdims) == (0:3, 11:20, -1:0, 2:4)

        gdims = @inferred NTuple{4,Dimensions.Dimension} InferenceObjects.generate_dims(
            x, :z; dims=(:c, :d), default_dims=(:chain, :draw)
        )
        @test gdims isa NTuple{4,Dim}
        @test Dimensions.name(gdims) === (:chain, :draw, :c, :d)
        @test Dimensions.index(gdims) == (0:3, 11:20, -1:0, 2:4)
    end

    @testset "array_to_dim_array" begin
        x = OffsetArray(randn(4, 10, 2, 3), 0:3, 11:20, -1:0, 2:4)
        da = @inferred DimArray InferenceObjects.array_to_dimarray(x, :x)
        @test da == x
        @test DimensionalData.name(da) === :x
        gdims = Dimensions.dims(da)
        @test gdims isa NTuple{4,Dim}
        @test Dimensions.name(gdims) === (:x_dim_1, :x_dim_2, :x_dim_3, :x_dim_4)
        @test Dimensions.index(gdims) == (0:3, 11:20, -1:0, 2:4)

        da = @inferred DimArray InferenceObjects.array_to_dimarray(x, :y; dims=(:a, :b))
        @test da == x
        @test DimensionalData.name(da) === :y
        gdims = Dimensions.dims(da)
        @test gdims isa NTuple{4,Dim}
        @test Dimensions.name(gdims) === (:a, :b, :y_dim_3, :y_dim_4)
        @test Dimensions.index(gdims) == (0:3, 11:20, -1:0, 2:4)

        da = @inferred DimArray InferenceObjects.array_to_dimarray(
            x, :z; dims=(:c, :d), default_dims=(:chain, :draw)
        )
        @test da == x
        @test DimensionalData.name(da) === :z
        gdims = Dimensions.dims(da)
        @test gdims isa NTuple{4,Dim}
        @test Dimensions.name(gdims) === (:chain, :draw, :c, :d)
        @test Dimensions.index(gdims) == (0:3, 11:20, -1:0, 2:4)
    end

    @testset "AsSlice" begin
        da = DimArray(randn(2), Dim{:a}(["foo", "bar"]))
        @test da[a=At("foo")] == da[1]
        da_sel = @inferred da[a=InferenceObjects.AsSlice(At("foo"))]
        @test da_sel isa DimArray
        @test Dimensions.dims(da_sel) == (Dim{:a}(["foo"]),)
        @test da_sel == da[a=At(["foo"])]

        da_sel = @inferred da[a=At(["foo", "bar"])]
        @test da_sel isa DimArray
        @test Dimensions.dims(da_sel) == Dimensions.dims(da)
        @test da_sel == da
    end

    @testset "index_to_indices" begin
        @test InferenceObjects.index_to_indices(1) == [1]
        @test InferenceObjects.index_to_indices(2) == [2]
        @test InferenceObjects.index_to_indices([2]) == [2]
        @test InferenceObjects.index_to_indices(1:10) === 1:10
        @test InferenceObjects.index_to_indices(At(1)) === InferenceObjects.AsSlice(At(1))
        @test InferenceObjects.index_to_indices(At(1)) === InferenceObjects.AsSlice(At(1))
    end
end
