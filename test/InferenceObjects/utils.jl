using ArviZ, Test
using ArviZ.InferenceObjects

@testset "package_version" begin
    @test InferenceObjects.package_version(ArviZ) isa VersionNumber
end
