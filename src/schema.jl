
struct InferenceDataSchemaError <: Exception
    msg::String
end

"""
    check_follows_schema(data::InferenceData; indices=false) -> Nothing
    check_follows_schema(data::Dataset; name=:dataset, required_dims=()) -> Nothing

Raise an [`InferenceDataSchemaError`](@ref) if `data` does not follow the
[InferenceData schema](https://python.arviz.org/en/v$(arviz_version())/schema/schema.html).

When applicable, if `indices=true`, values of indices are also checked for consistency
across groups.

Use [`follows_schema`](@ref) to return whether `data` follows the schema without raising an
error.
"""
check_follows_schema

function check_follows_schema(data::Dataset; name=:dataset, required_dims=())
    dims = Dimensions.dims(data)
    dim_names = Dimensions.name(dims)
    length(Dimensions.commondims(dims, dim_names)) == length(dims) ||
        throw(InferenceDataSchemaError("$name has non-unique dimension names: $dim_names"))
    # - groups that contain samples must contain the sample dimensions
    all(Dimensions.hasdim(data, required_dims)) || throw(
        InferenceDataSchemaError(
            "$name does not have the required dimensions: $required_dims"
        ),
    )
    # - variables must not share names with dimensions
    shared_var_dims = Dimensions.commondims(DimensionalData.keys(data), dims)
    if !isempty(shared_var_dims)
        throw(
            InferenceDataSchemaError(
                "$name has variables and dimensions with the same name: $(Dimensions.name(shared_var_dims))",
            ),
        )
    end
    metadata = DimensionalData.metadata(data)
    eltype(keys(metadata)) <: Symbol ||
        throw(InferenceDataSchemaError("$name has metadata with non-Symbol keys."))
    # - each group contains the attribute `:created_at`.
    haskey(metadata, :created_at) ||
        throw(InferenceDataSchemaError("$name has no `:created_at` entry in its metadata."))
    return nothing
end
function check_follows_schema(data::InferenceData; indices=false)
    # each group separately should follow the schema
    foreach(
        zip(groupnames(data), groups(data), _maybe_sample_dims(data))
    ) do (name, group, sample_dims)
        check_follows_schema(group; sample_dims, name)
    end
    sample_dims = DEFAULT_SAMPLE_DIMS
    # posterior-derived groups should share same sample dims/indices
    check_dims_match(_posterior_related_groups(data), sample_dims; indices)
    # prior-derived groups should share same sample dims/indices
    check_dims_match(_prior_related_groups(data), sample_dims; indices)
    # any parameters shared by prior and posterior should have the same non-sample dims/indices
    if hasgroup(data, :posterior) && hasgroup(data, :prior)
        var_dims = Dimensions.otherdims(
            Dimensions.commondims(data.posterior, data.prior), sample_dims
        )
        check_dims_match(NamedTuple{(:posterior, :prior)}(data), var_dims; indices)
    end
    # any dim names shared by log_likelihood, prior_predictive, posterior_predictive and
    # observed_data must share the same indices
    data_groups = _data_related_groups(data)
    if length(data_groups) > 1
        data_dims = Dimensions.otherdims(Dimensions.commondims(data_groups...), sample_dims)
        check_dims_match(data_groups, data_dims; indices)
    end
    return nothing
end

"""
    follows_schema(data; kwargs...) -> Bool

Return whether `data` follows the [InferenceData schema](https://python.arviz.org/en/v$(arviz_version())/schema/schema.html).

`kwargs` are passed to [`check_follows_schema`](@ref).

See [`check_follows_schema`](@ref) for informative error messages.
"""
function follows_schema(data; kwargs...)
    try
        check_follows_schema(data; kwargs...)
        return true
    catch e
        e isa InferenceDataSchemaError && return false
        throw(e)
    end
end

@generated function _maybe_sample_dims(::InferenceData{group_names}) where {group_names}
    return map(group_names) do name
        if name ∈ (
            :posterior,
            :posterior_predictive,
            :predictive,
            :sample_stats,
            :log_likelihood,
            :prior,
            :prior_predictive,
            :sample_stats_prior,
        )
            return DEFAULT_SAMPLE_DIMS
        else
            return ()
        end
    end
end

@generated function _filter_groups_type(
    ::InferenceData{groups}, ::Val{other}
) where {groups,other}
    shared = Tuple(intersect(groups, other))
    return NamedTuple{shared}
end

function _posterior_related_groups(idata::InferenceData)
    posterior_groups = (
        :posterior, :posterior_predictive, :sample_stats, :log_likelihood, :predictive
    )
    return _filter_groups_type(idata, Val(posterior_groups))(idata)
end

function _prior_related_groups(idata::InferenceData)
    prior_groups = (:prior, :prior_predictive, :sample_stats_prior)
    return _filter_groups_type(idata, Val(prior_groups))(idata)
end

function _data_related_groups(idata::InferenceData)
    data_groups = (
        :observed_data, :log_likelihood, :prior_predictive, :posterior_predictive
    )
    return _filter_groups_type(idata, Val(data_groups))(idata)
end

function check_dims_match(groups, dims; indices::Bool=true)
    isempty(groups) && return nothing
    try
        # dims = Dimensions.commondims(first(groups), _dims)
        # if length(dims) != length(_dims)
        #     dims_missing = setdiff(Dimensions.name(_dims), Dimensions.name(dims))
        #     throw(ErrorException("dimensions missing $(dims_missing)"))
        # end
        comparesomedims(groups...; dims, val=indices)
    catch e
        if e isa DimensionMismatch
            throw(
                InferenceDataSchemaError(
                    "Dimension mismatch in groups $(keys(groups)): $(e.msg)"
                ),
            )
        else
            throw(e)
        end
    end
    return nothing
end

function comparesomedims(datasets::Dataset...; dims, val=false)
    sub_dims = map(datasets) do ds
        Dimensions.sortdims(Dimensions.commondims(ds, dims), dims)
    end
    sub_dims_ref = first(sub_dims)
    for _dims in Iterators.drop(sub_dims, 1)
        Dimensions.comparedims(sub_dims_ref, _dims; val)
    end
    return nothing
end
