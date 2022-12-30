using ArviZ, DimensionalData, Test

@testset "extract" begin
    idata = random_data()
    post = extract(idata, :posterior; combined=false)
    for k in keys(idata.posterior)
        @test haskey(post, k)
        @test post[k] ≈ idata.posterior[k]
        dims = DimensionalData.dims(post)
        dims_exp = DimensionalData.dims(idata.posterior)
        @test DimensionalData.name(dims) === DimensionalData.name(dims_exp)
        @test DimensionalData.index(dims) == DimensionalData.index(dims_exp)
    end
    prior = extract(idata, :prior; combined=false)
    for k in keys(idata.prior)
        @test haskey(prior, k)
        @test prior[k] ≈ idata.prior[k]
        dims = DimensionalData.dims(prior)
        dims_exp = DimensionalData.dims(idata.prior)
        @test DimensionalData.name(dims) === DimensionalData.name(dims_exp)
        @test DimensionalData.index(dims) == DimensionalData.index(dims_exp)
    end
end

@testset "concat" begin
    data = random_data()
    idata1 = InferenceData(; posterior=data.posterior)
    idata2 = InferenceData(; prior=data.prior)
    new_idata1 = concat(idata1, idata2)
    new_idata2 = InferenceData(; posterior=data.posterior, prior=data.prior)
    test_idata_approx_equal(new_idata1, new_idata2)
end
