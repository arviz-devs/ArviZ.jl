@testset "utils" begin
    @testset "replacemissing" begin
        @test isnan(ArviZ.replacemissing(missing))
        x = 1.0
        @test ArviZ.replacemissing(x) === x
        x = [1.0, 2.0, 3.0]
        @test ArviZ.replacemissing(x) === x
        x = [1.0, missing, 2.0]
        x2 = ArviZ.replacemissing(x)
        @test eltype(x2) <: Real
        @test x2[1] == 1.0
        @test isnan(x2[2])
        @test x2[3] == 2.0
        x = Any[1.0, missing, 2.0]
        x2 = ArviZ.replacemissing(x)
        @test eltype(x2) <: Real
        @test x2[1] == 1.0
        @test isnan(x2[2])
        @test x2[3] == 2.0

        x = [[1.0, missing, 2.0]]
        x2 = ArviZ.replacemissing([[1.0, missing, 2.0]])
        @test x2 isa Vector
        @test x2[1] isa Vector
        @test x2[1][1] == 1.0
        @test isnan(x2[1][2])
        @test x2[1][3] == 2.0
    end

    @testset "convert_to_eltypes" begin
        data1 = Dict("x" => rand((0, 1), 10), "y" => rand(10))
        data1_format = ArviZ.convert_to_eltypes(data1, Dict("x" => Bool))
        keys(data1) == keys(data1_format)
        @test data1_format["x"] == data1["x"]
        @test eltype(data1_format["x"]) === Bool
        @test data1_format["y"] == data1["y"]
        @test eltype(data1_format["y"]) === eltype(data1["y"])

        data2 = (x=rand(1:3, 10), y=randn(10))
        data2_format = ArviZ.convert_to_eltypes(data2, (; x=Int))
        propertynames(data2) == propertynames(data2_format)
        @test data2_format.x == data2.x
        @test eltype(data2_format.x) === Int
        @test data2_format.y == data2.y
        @test eltype(data2_format.y) === eltype(data2.y)
    end
end
