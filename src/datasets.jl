"""
    load_arviz_data(dataset = nothing; data_home = nothing)

Load a local or remote pre-made `dataset` as an `InferenceData`, saving remote
datasets to `data_home`.

The directory to save to can also be set with the environement variable
`ARVIZ_HOME`. The checksum of the dataset is checked against a hardcoded value
to watch for data corruption.
"""
function load_arviz_data(dataset; data_home = nothing)
    data = arviz.load_arviz_data(dataset; data_home = data_home)
    return InferenceData(data)
end

"""
    load_arviz_data()

Get a list of all available local or remote pre-made datasets.
"""
load_arviz_data() = arviz.load_arviz_data()
