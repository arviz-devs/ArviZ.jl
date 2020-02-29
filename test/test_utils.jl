import DataFrames
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

    @testset "todataframes" begin
        @testset "pandas.DataFrame -> DataFrames.DataFrame" begin
            colnames = [:a, :b, :c]
            index = ["d", "e"]
            rowvals = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            pdf = pandas.DataFrame(rowvals; columns = colnames, index = index)
            df = ArviZ.todataframes(pdf; index_name = :i)
            @test df isa DataFrames.DataFrame
            @test df == DataFrames.DataFrame([
                :i => ["d", "e"],
                :a => [1.0, 4.0],
                :b => [2.0, 5.0],
                :c => [3.0, 6.0],
            ])
            @test df == ArviZ.todataframes(pdf; index_name = :i)
        end

        @testset "pandas.Series -> DataFrames.DataFrame" begin
            ps = pandas.Series([1.0, 2.0, 3.0], [:a, :b, :c])
            df2 = ArviZ.todataframes(ps)
            @test df2 isa DataFrames.DataFrame
            @test df2 == DataFrames.DataFrame([:a => [1.0], :b => [2.0], :c => [3.0]])
            @test df2 == ArviZ.todataframes(ps)
        end
    end

    @testset "topandas" begin
        @testset "DataFrames.DataFrame -> pandas.DataFrame" begin
            colnames = [:a, :b, :c]
            index = ["d", "e"]
            rowvals = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            df = DataFrames.DataFrame([
                :i => ["d", "e"],
                :a => [1.0, 4.0],
                :b => [2.0, 5.0],
                :c => [3.0, 6.0],
            ])
            pdf = ArviZ.topandas(Val(:DataFrame), df; index_name = :i)
            @test pyisinstance(pdf, pandas.DataFrame)
            pdf_exp = pandas.DataFrame(rowvals; columns = colnames, index = index)
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
            idata = load_arviz_data("centered_eight")
            df = loo(idata)
            elpd_data = ArviZ.arviz.loo(idata)
            @test all(df == ArviZ.todataframes(elpd_data))
            ps = ArviZ.topandas(Val(:ELPDData), df)
            @test pyisinstance(ps, ArviZ.arviz.stats.ELPDData)
            @test py"($(ps) == $(elpd_data)).all()"
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
end
