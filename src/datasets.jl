"""
    load_arviz_data(dataset; data_home = nothing)

Load a local or remote pre-made `dataset` as an `InferenceData`, saving remote
datasets to `data_home`.

The directory to save to can also be set with the environement variable
`ARVIZ_HOME`. The checksum of the dataset is checked against a hardcoded value
to watch for data corruption.

    load_arviz_data()

Get a list of all available local or remote pre-made datasets.
"""
@forwardfun load_arviz_data
