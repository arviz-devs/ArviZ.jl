"""
    InferenceData{group_names,group_types}

Container for inference data storage using DimensionalData.

This object implements the [InferenceData schema](https://python.arviz.org/en/v$(arviz_version())/schema/schema.html).

Internally, groups are stored in a `NamedTuple`, which can be accessed using
`parent(::InferenceData)`.

# Constructors

    InferenceData(groups::NamedTuple)
    InferenceData(; groups...)

Construct an inference data from either a `NamedTuple` or keyword arguments of groups.

Groups must be [`Dataset`](@ref) objects.

Instead of directly creating an `InferenceData`, use the exported `from_xyz` functions or
[`convert_to_inference_data`](@ref).
"""
struct InferenceData{group_names,group_types<:Tuple{Vararg{Dataset}}}
    groups::NamedTuple{group_names,group_types}
    function InferenceData(
        groups::NamedTuple{group_names,<:Tuple{Vararg{Dataset}}}
    ) where {group_names}
        group_names_ordered = _reorder_group_names(Val{group_names}())
        groups_ordered = NamedTuple{group_names_ordered}(groups)
        return new{group_names_ordered,typeof(values(groups_ordered))}(groups_ordered)
    end
end
InferenceData(; kwargs...) = InferenceData(NamedTuple(kwargs))
InferenceData(data::InferenceData) = data

Base.parent(data::InferenceData) = getfield(data, :groups)

"""
    convert(::Type{InferenceData}, obj)

Convert `obj` to an `InferenceData`.

`obj` can be any type for which [`convert_to_inference_data`](@ref) is defined.
"""
Base.convert(::Type{InferenceData}, obj) = convert_to_inference_data(obj)
Base.convert(::Type{InferenceData}, obj::InferenceData) = obj
Base.convert(::Type{NamedTuple}, data::InferenceData) = NamedTuple(data)
NamedTuple(data::InferenceData) = parent(data)

# these 3 interfaces ensure InferenceData behaves like a NamedTuple

# properties interface
"""
    propertynames(data::InferenceData) -> Tuple{Symbol}

Get names of groups
"""
Base.propertynames(data::InferenceData) = propertynames(parent(data))

"""
    getproperty(data::InferenceData, name::Symbol) -> Dataset

Get group with the specified `name`.
"""
Base.getproperty(data::InferenceData, k::Symbol) = getproperty(parent(data), k)

# indexing interface
"""
    Base.getindex(data::InferenceData, groups::Symbol; coords...) -> Dataset
    Base.getindex(data::InferenceData, groups; coords...) -> InferenceData

Return a new `InferenceData` containing the specified groups sliced to the specified coords.

`coords` specifies a dimension name mapping to an index, a `DimensionalData.Selector`, or
an `IntervalSets.AbstractInterval`.

If one or more groups lack the specified dimension, a warning is raised but can be ignored.
All groups that contain the dimension must also contain the specified indices, or an
exception will be raised.

# Examples

Select data from all groups for just the specified schools.

```@repl getindex
julia> using ArviZ, DimensionalData

julia> idata = load_example_data("centered_eight");

julia> idata_sel = idata[school=At(["Choate", "Deerfield"])]
InferenceData with groups:
  > posterior
  > posterior_predictive
  > sample_stats
  > prior
  > observed_data

julia> idata_sel.posterior
Dataset with dimensions:
  Dim{:chain} Sampled 0:3 ForwardOrdered Regular Points,
  Dim{:draw} Sampled 0:499 ForwardOrdered Regular Points,
  Dim{:school} Categorical String[Choate, Deerfield] Unordered
and 3 layers:
  :mu    Float64 dims: Dim{:chain}, Dim{:draw} (4×500)
  :theta Float64 dims: Dim{:chain}, Dim{:draw}, Dim{:school} (4×500×2)
  :tau   Float64 dims: Dim{:chain}, Dim{:draw} (4×500)

with metadata OrderedCollections.OrderedDict{Symbol, Any} with 3 entries:
  :created_at                => "2019-06-21T17:36:34.398087"
  :inference_library_version => "3.7"
  :inference_library         => "pymc3"
```

Select data from just the posterior, returning a `Dataset` if the indices index more than
one element from any of the variables:

```@repl getindex
julia> idata[:observed_data, school=At(["Choate"])]
Dataset with dimensions:
  Dim{:school} Categorical String[Choate] Unordered
and 1 layer:
  :obs Float64 dims: Dim{:school} (1)

with metadata OrderedCollections.OrderedDict{Symbol, Any} with 3 entries:
  :created_at                => "2019-06-21T17:36:34.491909"
  :inference_library_version => "3.7"
  :inference_library         => "pymc3"
```

Note that if a single index is provided, the behavior is still to slice so that the
dimension is preserved.
"""
Base.getindex(data::InferenceData, groups...; kwargs...)
function Base.getindex(data::InferenceData, k::Symbol; kwargs...)
    ds = parent(data)[k]
    isempty(kwargs) && return ds
    return getindex(ds; kwargs...)
end
function Base.getindex(data::InferenceData, i::Int; kwargs...)
    ds = parent(data)[i]
    isempty(kwargs) && return ds
    return getindex(ds; kwargs...)
end
function Base.getindex(data::InferenceData, ks; kwargs...)
    data_new = InferenceData(parent(data)[ks])
    isempty(kwargs) && return data_new
    return getindex(data_new; kwargs...)
end
function Base.getindex(data::InferenceData; kwargs...)
    # if a single index is requested, then the return type of each group
    # will be a `Dataset` if the group has other dimensions or `NamedTuple`
    # if it has no other dimensions.
    # So we promote to an array of indices
    new_kwargs = map(_index_to_indices, NamedTuple(kwargs))
    groups = map(parent(data)) do ds
        return getindex(ds; new_kwargs...)
    end
    return InferenceData(groups)
end

"""
    Base.setindex(data::InferenceData, group::Dataset, name::Symbol) -> InferenceData

Create a new `InferenceData` containing the `group` with the specified `name`.

If a group with `name` is already in `data`, it is replaced.
"""
function Base.setindex(data::InferenceData, v, k::Symbol)
    return InferenceData(Base.setindex(parent(data), v, k))
end

# iteration interface
Base.keys(data::InferenceData) = keys(parent(data))
Base.haskey(data::InferenceData, k::Symbol) = haskey(parent(data), k)
Base.values(data::InferenceData) = values(parent(data))
Base.pairs(data::InferenceData) = pairs(parent(data))
Base.length(data::InferenceData) = length(parent(data))
Base.iterate(data::InferenceData, i...) = iterate(parent(data), i...)
Base.eltype(data::InferenceData) = eltype(parent(data))

function Base.show(io::IO, ::MIME"text/plain", data::InferenceData)
    print(io, "InferenceData with groups:")
    prefix = "\n  > "
    for name in groupnames(data)
        print(io, prefix, name)
    end
    return nothing
end
function Base.show(io::IO, mime::MIME"text/html", data::InferenceData)
    show(io, mime, HTML("<div>InferenceData"))
    for (name, group) in pairs(groups(data))
        show(io, mime, HTML("""
        <details>
        <summary>$name</summary>
        <pre><code>$(sprint(show, "text/plain", group))</code></pre>
        </details>
        """))
    end
    return show(io, mime, HTML("</div>"))
end

"""
    groups(data::InferenceData)

Get the groups in `data` as a named tuple mapping symbols to [`Dataset`](@ref)s.
"""
groups(data::InferenceData) = parent(data)

"""
    groupnames(data::InferenceData)

Get the names of the groups (datasets) in `data` as a tuple of symbols.
"""
groupnames(data::InferenceData) = keys(groups(data))

"""
    hasgroup(data::InferenceData, name::Symbol) -> Bool

Return `true` if a group with name `name` is stored in `data`.
"""
hasgroup(data::InferenceData, name::Symbol) = haskey(data, name)

_index_to_indices(i) = i
_index_to_indices(i::Int) = [i]
_index_to_indices(sel::Dimensions.Selector) = AsSlice(sel)

@generated function _reorder_group_names(::Val{names}) where {names}
    return Tuple(sort(collect(names); by=k -> SUPPORTED_GROUPS_DICT[k]))
end

@generated _keys_and_types(::NamedTuple{keys,types}) where {keys,types} = (keys, types)
