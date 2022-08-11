using ArviZ.InferenceObjects, Test
using ArviZ.InferenceObjects: package_version

@testset "package_version" begin
    @test package_version(ArviZ) isa VersionNumber
end
