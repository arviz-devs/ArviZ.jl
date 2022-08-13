using ArviZ, Test
using ArviZ.InferenceObjects

@testset "utils" begin
    @testset "package_version" begin
        @test InferenceObjects.package_version(ArviZ) isa VersionNumber
    end

    @testset "rekey" begin
        orig = (x=3, y=4, z=5)
        keymap = (x=:y, y=:a)
        @testset "NamedTuple" begin
            new = @inferred NamedTuple ArviZ.rekey(orig, keymap)
            @test new isa NamedTuple
            @test new == (y=3, a=4, z=5)
        end
        @testset "Dict" begin
            orig_dict = Dict(pairs(orig))
            new = @inferred ArviZ.rekey(orig_dict, keymap)
            @test new isa typeof(orig_dict)
            @test new == Dict(:y => 3, :a => 4, :z => 5)
        end
    end
end
