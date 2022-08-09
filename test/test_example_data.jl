using ArviZ, Test

@testset "load_example_data" begin
    names = [
        "centered_eight",
        "classification10d",
        "classification1d",
        "glycan_torsion_angles",
        "non_centered_eight",
        "radon",
        "regression10d",
        "regression1d",
        "rugby",
    ]
    datasets = load_example_data()
    @test datasets isa Dict{String,ArviZ.AbstractFileMetadata}
    @test issetequal(keys(datasets), names)
    for name in names
        idata = load_example_data(name)
        @test idata isa InferenceData
        @test ArviZ.hasgroup(idata, :posterior)
    end
end

