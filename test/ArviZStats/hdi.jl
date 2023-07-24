using ArviZ
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
        end
    end
end
