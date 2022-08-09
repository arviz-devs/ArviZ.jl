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

@testset "load_arviz_data" begin
    data = @test_deprecated load_arviz_data("centered_eight")
    datasets = @test_deprecated load_arviz_data()
    @test datasets isa Dict
    mktempdir() do data_home
        @test_deprecated load_arviz_data("rugby", data_home)
        @test readdir(data_home) == ["rugby.nc"]
    end
end
