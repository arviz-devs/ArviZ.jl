using ArviZ, Test

@testset "example data" begin
    @testset "LocalFileMetadata" begin
        md = ArviZ.LocalFileMetadata(; name="foo", filename="bar.nc")
        @test md.name == "foo"
        @test md.filename == "bar.nc"
        @test md.description === nothing
        @test startswith(sprint(show, "text/plain", md), "foo\n===\n\nlocal:")

        md = ArviZ.LocalFileMetadata(; name="foo", filename="bar.nc", description="desc")
        @test startswith(sprint(show, "text/plain", md), "foo\n===\n\ndesc\n\nlocal:")
    end

    @testset "RemoteFileMetadata" begin
        md = ArviZ.RemoteFileMetadata(; name="foo", filename="bar.nc", url="http://baq.baz")
        @test md.name == "foo"
        @test md.filename == "bar.nc"
        @test md.url == "http://baq.baz"
        @test md.description === nothing
        @test md.checksum === nothing
        @test sprint(show, "text/plain", md) == "foo\n===\n\nremote: http://baq.baz"

        md = ArviZ.RemoteFileMetadata(;
            name="foo", filename="bar.nc", url="http://baq.baz", description="desc"
        )
        @test sprint(show, "text/plain", md) == "foo\n===\n\ndesc\n\nremote: http://baq.baz"
    end

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
        @test issubset(keys(datasets), names)
        for name in keys(datasets)
            idata = load_example_data(name)
            @test idata isa InferenceData
            @test ArviZ.hasgroup(idata, :posterior)
        end
        @test_throws ArgumentError load_example_data("test_absent_dataset")
    end

    @testset "load_arviz_data" begin
        data = @test_deprecated load_arviz_data("centered_eight")
        datasets = @test_deprecated load_arviz_data()
        @test datasets isa Dict
        VERSION >= v"1.8" && mktempdir() do data_home
            @test_deprecated load_arviz_data("rugby", data_home)
            @test readdir(data_home) == ["rugby.nc"]
        end
    end
end
