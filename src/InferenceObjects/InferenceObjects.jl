module InferenceObjects

using DimensionalData: DimensionalData, Dimensions

export Dataset, InferenceData
export convert_to_dataset, convert_to_inference_data, namedtuple_to_dataset

include("dataset.jl")
include("inference_data.jl")

end # module
