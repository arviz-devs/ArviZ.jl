using DataFrames: DataFrames
using PyCall, PyPlot

pandas = ArviZ.pandas

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

    @testset "frompytype" begin
        x = 1.0
        @test ArviZ.frompytype(x) === x
        x2 = PyObject(x)
        @test ArviZ.frompytype(x2) == x
        @test ArviZ.frompytype([x2]) == [x]
        @test ArviZ.frompytype(Any[x2]) == [x]
        @test eltype(ArviZ.frompytype(Any[x2])) <: Real
        @test ArviZ.frompytype([[x2]]) == [[x]]
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

    @testset "topandas" begin
        @testset "DataFrames.DataFrame -> pandas.DataFrame" begin
            columns = [:a, :b, :c]
            index = ["d", "e"]
            rowvals = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            df = DataFrames.DataFrame([
                :i => ["d", "e"], :a => [1.0, 4.0], :b => [2.0, 5.0], :c => [3.0, 6.0]
            ])
            pdf = ArviZ.topandas(Val(:DataFrame), df; index_name=:i)
            @test pyisinstance(pdf, pandas.DataFrame)
            pdf_exp = pandas.DataFrame(rowvals; columns, index)
            @test py"($(pdf) == $(pdf_exp)).all().all()"
        end

        @testset "DataFrames.DataFrame -> pandas.Series" begin
            df2 = DataFrames.DataFrame([:a => [1.0], :b => [2.0], :c => [3.0]])
            ps = ArviZ.topandas(Val(:Series), df2)
            @test pyisinstance(ps, pandas.Series)
            ps_exp = pandas.Series([1.0, 2.0, 3.0], [:a, :b, :c])
            @test py"($(ps) == $(ps_exp)).all()"
        end

        @testset "DataFrames.DataFrame -> ELPDData" begin
            idata = load_example_data("centered_eight")
            df = loo(idata; pointwise=false)
            elpd_data = ArviZ.arviz.loo(idata; pointwise=false)
            @test all(df == ArviZ.todataframes(elpd_data))
            ps = ArviZ.topandas(Val(:ELPDData), df)
            @test pyisinstance(ps, ArviZ.arviz.stats.ELPDData)
            @test py"($(ps) == $(elpd_data)).all()"
        end
    end

    @testset "todataframes" begin
        @testset "pandas.DataFrame -> DataFrames.DataFrame" begin
            columns = [:a, :b, :c]
            index = ["d", "e"]
            rowvals = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            pdf = pandas.DataFrame(rowvals; columns, index)
            df = ArviZ.todataframes(pdf; index_name=:i)
            @test df isa DataFrames.DataFrame
            @test df == DataFrames.DataFrame([
                :i => ["d", "e"], :a => [1.0, 4.0], :b => [2.0, 5.0], :c => [3.0, 6.0]
            ])
            @test df == ArviZ.todataframes(pdf; index_name=:i)
        end

        @testset "pandas.Series -> DataFrames.DataFrame" begin
            ps = pandas.Series([1.0, 2.0, 3.0], [:a, :b, :c])
            df2 = ArviZ.todataframes(ps)
            @test df2 isa DataFrames.DataFrame
            @test df2 == DataFrames.DataFrame([:a => [1.0], :b => [2.0], :c => [3.0]])
            @test df2 == ArviZ.todataframes(ps)
        end
    end

    @testset "styles" begin
        @test ArviZ.styles() isa AbstractArray{String}
        @test "arviz-darkgrid" ∈ ArviZ.styles()
        @test ArviZ.styles() == ArviZ.arviz.style.available
    end

    @testset "use_style" begin
        ArviZ.use_style("arviz-darkgrid")
        ArviZ.use_style("default")
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

    @testset "package_version" begin
        @test ArviZ.package_version(ArviZ) isa VersionNumber
        @test ArviZ.package_version(PyCall) isa VersionNumber
    end
end
