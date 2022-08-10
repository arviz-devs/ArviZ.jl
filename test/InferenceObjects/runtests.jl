using ArviZ.InferenceObjects, Test

@testset "InferenceObjects" begin
    include("../helpers.jl")
    include("utils.jl")
    include("dataset.jl")
    include("inference_data.jl")
    include("convert_dataset.jl")
    include("convert_inference_data.jl")
end
