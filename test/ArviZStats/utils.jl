using Test
using ArviZ
using ArviZ.ArviZStats

@testset "utils" begin
    # tests for all utilities in this file
    @testset "_get_log_likelihood" begin
        ndraws = 100
        nchains = 4
        nparams = 3
        x = randn(ndraws, nchains, nparams)
        log_likelihood = convert_to_dataset((; x))
        @test ArviZStats._get_log_likelihood(log_likelihood) == x
        @test ArviZStats._get_log_likelihood(log_likelihood; var_name=:x) == x
        @test_throws Exception ArviZStats._get_log_likelihood(log_likelihood; var_name=:y)
        idata = InferenceData(; log_likelihood)
        @test ArviZStats._get_log_likelihood(idata) == x
        @test ArviZStats._get_log_likelihood(idata; var_name=:x) == x
        @test_throws Exception ArviZStats._get_log_likelihood(idata; var_name=:y)

        y = randn(ndraws, nchains)
        log_likelihood = convert_to_dataset((; x, y))
        @test_throws Exception ArviZStats._get_log_likelihood(log_likelihood)
        @test ArviZStats._get_log_likelihood(log_likelihood; var_name=:x) == x
        @test ArviZStats._get_log_likelihood(log_likelihood; var_name=:y) == y

        idata = InferenceData(; log_likelihood)
        @test_throws Exception ArviZStats._get_log_likelihood(idata)
        @test ArviZStats._get_log_likelihood(idata; var_name=:x) == x
        @test ArviZStats._get_log_likelihood(idata; var_name=:y) == y

        # test old InferenceData versions
        sample_stats = convert_to_dataset((; lp=randn(ndraws, nchains), log_likelihood=x))
        idata = InferenceData(; sample_stats)
        @test ArviZStats._get_log_likelihood(idata) == x

        sample_stats = convert_to_dataset((; lp=randn(ndraws, nchains), log_like=x))
        idata = InferenceData(; sample_stats)
        @test_throws Exception ArviZStats._get_log_likelihood(idata)
        @test ArviZStats._get_log_likelihood(idata; var_name=:log_like) == x
    end
end
