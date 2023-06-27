using Test
using ArviZ
using ArviZ.ArviZStats

@testset "utils" begin
    # tests for all utilities in this file
    @testset "log_likelihood" begin
        ndraws = 100
        nchains = 4
        nparams = 3
        x = randn(ndraws, nchains, nparams)
        log_like = convert_to_dataset((; x))
        @test ArviZStats.log_likelihood(log_like) == x
        @test ArviZStats.log_likelihood(log_like, :x) == x
        @test_throws Exception ArviZStats.log_likelihood(log_like, :y)
        idata = InferenceData(; log_likelihood=log_like)
        @test ArviZStats.log_likelihood(idata) == x
        @test ArviZStats.log_likelihood(idata, :x) == x
        @test_throws Exception ArviZStats.log_likelihood(idata, :y)

        y = randn(ndraws, nchains)
        log_like = convert_to_dataset((; x, y))
        @test_throws Exception ArviZStats.log_likelihood(log_like)
        @test ArviZStats.log_likelihood(log_like, :x) == x
        @test ArviZStats.log_likelihood(log_like, :y) == y

        idata = InferenceData(; log_likelihood=log_like)
        @test_throws Exception ArviZStats.log_likelihood(idata)
        @test ArviZStats.log_likelihood(idata, :x) == x
        @test ArviZStats.log_likelihood(idata, :y) == y

        # test old InferenceData versions
        sample_stats = convert_to_dataset((; lp=randn(ndraws, nchains), log_likelihood=x))
        idata = InferenceData(; sample_stats)
        @test ArviZStats.log_likelihood(idata) == x

        sample_stats = convert_to_dataset((; lp=randn(ndraws, nchains), log_like=x))
        idata = InferenceData(; sample_stats)
        @test_throws Exception ArviZStats.log_likelihood(idata)
        @test ArviZStats.log_likelihood(idata, :log_like) == x
    end
end
