@testset "rcParams" begin
    @test ArviZ.arviz.rcparams.rcParams["data.index_origin"] == 1
end
