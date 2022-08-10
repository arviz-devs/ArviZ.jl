module InferenceObjects

using DimensionalData: DimensionalData, Dimensions

export Dataset, InferenceData
export convert_to_dataset, convert_to_inference_data, namedtuple_to_dataset

include("utils.jl")
include("dataset.jl")
include("inference_data.jl")
include("convert_dataset.jl")
include("convert_inference_data.jl")

end # module
