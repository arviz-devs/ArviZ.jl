using ArviZ
using DimensionalData
using OffsetArrays
using Statistics
using Test

@testset "hdi/hdi!" begin
    @testset "AbstractVector" begin
        @testset for n in (10, 100, 1_000),
            prob in (eps(), 0.5, 0.73, 0.96, 1.0),
            T in (Float32, Float64, Int64)

            x = T <: Integer ? rand(T(1):T(30), n) : randn(T, n)
            r = @inferred hdi(x; prob)
            @test r isa NamedTuple{(:lower, :upper),NTuple{2,T}}
            l, u = r
            interval_length = ceil(Int, prob * n)
            if T <: Integer
                @test sum(x -> l ≤ x ≤ u, x) ≥ interval_length
            else
                @test sum(x -> l ≤ x ≤ u, x) == interval_length
            end
            xsort = sort(x)
            lind = 1:(n - interval_length + 1)
            uind = interval_length:n
            @assert all(uind .- lind .+ 1 .== interval_length)
            @test minimum(xsort[uind] - xsort[lind]) ≈ u - l

            @test hdi!(copy(x); prob) == r
        end
    end

    @testset "edge cases and errors" begin
        @testset "NaNs returned if contains NaNs" begin
            x = randn(1000)
            x[3] = NaN
            @test isequal(hdi(x), (lower=NaN, upper=NaN))
        end

        @testset "errors for empty array" begin
            x = Float64[]
            @test_throws ArgumentError hdi(x)
        end

        @testset "errors for 0-dimensional array" begin
            x = fill(1.0)
            @test_throws ArgumentError hdi(x)
        end

        @testset "test errors when prob is not in (0, 1]" begin
            x = randn(1_000)
            @testset for prob in (0, -0.1, 1.1, NaN)
                @test_throws DomainError hdi(x; prob)
            end
        end
    end

    @testset "AbstractArray consistent with AbstractVector" begin
        @testset for sz in ((100, 2), (100, 2, 3), (100, 2, 3, 4)),
            prob in (0.72, 0.81),
            T in (Float32, Float64, Int64)

            x = T <: Integer ? rand(T(1):T(30), sz) : randn(T, sz)
            r = @inferred hdi(x; prob)
            if ndims(x) == 2
                @test r isa NamedTuple{(:lower, :upper),NTuple{2,T}}
                @test r == hdi(vec(x); prob)
            else
                @test r isa NamedTuple{(:lower, :upper),NTuple{2,Array{T,length(sz) - 2}}}
                r_slices = dropdims(
                    mapslices(x -> hdi(x; prob), x; dims=(1, 2)); dims=(1, 2)
                )
                @test r.lower == first.(r_slices)
                @test r.upper == last.(r_slices)
            end

            @test hdi!(copy(x); prob) == r
        end
    end

    @testset "OffsetArray" begin
        @testset for n in (100, 1_000), prob in (0.732, 0.864), T in (Float32, Float64)
            x = randn(T, (n, 2, 3, 4))
            xoff = OffsetArray(x, (-1, 2, -3, 4))
            r = hdi(x; prob)
            roff = @inferred hdi(xoff; prob)
            @test roff isa NamedTuple{(:lower, :upper),<:NTuple{2,OffsetMatrix{T}}}
            @test axes(roff.lower) == (axes(xoff, 3), axes(xoff, 4))
            @test axes(roff.upper) == (axes(xoff, 3), axes(xoff, 4))
            @test collect(roff.lower) == r.lower
            @test collect(roff.upper) == r.upper
        end
    end

    @testset "Dataset/InferenceData" begin
        nt = (x=randn(1000, 3), y=randn(1000, 3, 4), z=randn(1000, 3, 4, 2))
        posterior = convert_to_dataset(nt)
        posterior_perm = convert_to_dataset((
            x=permutedims(posterior.x),
            y=permutedims(posterior.y, (3, 2, 1)),
            z=permutedims(posterior.z, (3, 2, 4, 1)),
        ))
        idata = InferenceData(; posterior)
        @testset for prob in (0.76, 0.93)
            r1 = @inferred hdi(posterior; prob)
            r1_perm = hdi(posterior_perm; prob)
            for k in (:x, :y, :z)
                rk = hdi(posterior[k]; prob)
                @test r1[k][hdi_bound=At(:lower)] == rk.lower
                @test r1[k][hdi_bound=At(:upper)] == rk.upper
                # equality check is safe because these are always data values
                @test r1_perm[k][hdi_bound=At(:lower)] == rk.lower
                @test r1_perm[k][hdi_bound=At(:upper)] == rk.upper
            end
            r2 = @inferred hdi(idata; prob)
            @test r1 == r2
        end
    end
end
