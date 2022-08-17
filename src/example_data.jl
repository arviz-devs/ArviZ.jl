const DEPS_DATA_DIR = joinpath(pkgdir(ArviZ), "deps", "data")
const EXAMPLE_DATA_DIR = joinpath(DEPS_DATA_DIR, "example_data")
const LOCAL_DATA_DIR = joinpath(EXAMPLE_DATA_DIR, "data")
const LOCAL_DATA_FILE = joinpath(EXAMPLE_DATA_DIR, "data_local.json")
const REMOTE_DATA_FILE = joinpath(EXAMPLE_DATA_DIR, "data_remote.json")
const DATA_HTTP_PROTOCOL = "https"

abstract type AbstractFileMetadata end

function StructTypes.StructType(::Type{T}) where {T<:AbstractFileMetadata}
    return StructTypes.Struct()
end

Base.@kwdef struct LocalFileMetadata <: AbstractFileMetadata
    name::String
    filename::String
    description::Union{String,Nothing} = nothing
end

function Base.show(io::IO, ::MIME"text/plain", md::LocalFileMetadata)
    println(io, md.name)
    println(io, repeat("=", length(md.name)))
    println(io)
    if md.description !== nothing
        println(io, md.description)
        println(io)
    end
    print(io, "local: ", joinpath(LOCAL_DATA_DIR, md.filename))
    return nothing
end
Base.@kwdef struct RemoteFileMetadata <: AbstractFileMetadata
    name::String
    filename::String
    url::String
    checksum::Union{String,Nothing} = Nothing
    description::Union{String,Nothing} = Nothing
end

function Base.show(io::IO, ::MIME"text/plain", md::RemoteFileMetadata)
    println(io, md.name)
    println(io, repeat("=", length(md.name)))
    println(io)
    if md.description !== nothing
        println(io, md.description)
        println(io)
    end
    print(io, "remote: ", md.url)
    return nothing
end

const LOCAL_EXAMPLE_DATA = Dict(
    md.name => md for md in JSON3.read(read(LOCAL_DATA_FILE), Vector{LocalFileMetadata})
)

const REMOTE_EXAMPLE_DATA = Dict(
    md.name => md for md in JSON3.read(read(REMOTE_DATA_FILE), Vector{RemoteFileMetadata})
)

"""
    load_example_data(name; kwargs...) -> InferenceData
    load_example_data() -> Dict{String,AbstractFileMetadata}

Load a local or remote pre-made dataset.

`kwargs` are forwarded to [`from_netcdf`](@ref).

Pass no parameters to get a `Dict` listing all available datasets.

Data files are handled by DataDeps.jl.
A file is downloaded only when it is requested and then cached for future use.

# Examples

```jldoctest
julia> keys(load_example_data())
KeySet for a Dict{String, ArviZ.AbstractFileMetadata} with 9 entries. Keys:
  "centered_eight"
  "radon"
  "glycan_torsion_angles"
  "rugby"
  "non_centered_eight"
  "regression10d"
  "classification1d"
  "classification10d"
  "regression1d"

julia> load_example_data("centered_eight")
InferenceData with groups:
  > posterior
  > posterior_predictive
  > sample_stats
  > prior
  > observed_data
```
"""
function load_example_data end
function load_example_data(name::AbstractString; kwargs...)
    if haskey(LOCAL_EXAMPLE_DATA, name)
        path = joinpath(LOCAL_DATA_DIR, LOCAL_EXAMPLE_DATA[name].filename)
    elseif haskey(REMOTE_EXAMPLE_DATA, name)
        metadata = REMOTE_EXAMPLE_DATA[name]
        filename = joinpath(metadata.name, metadata.filename)
        # none of our downloads require license agreements, so we always agree to the download
        path = withenv("DATADEPS_ALWAYS_ACCEPT" => true) do
            return DataDeps.resolve(filename, @__FILE__)
        end
    else
        throw(
            ArgumentError(
                "Dataset $name not found. Valid datasets are: $(keys(EXAMPLE_DATA))"
            ),
        )
    end
    return from_netcdf(path; kwargs...)
end
load_example_data() = merge(LOCAL_EXAMPLE_DATA, REMOTE_EXAMPLE_DATA)

function _register_data(md::LocalFileMetadata)
    return DataDeps.register(DataDeps.ManualDataDep(md.name, md.description))
end

function _register_data(md::RemoteFileMetadata)
    url = replace(md.url, "http" => DATA_HTTP_PROTOCOL)
    return DataDeps.register(DataDeps.DataDep(md.name, md.description, url, md.checksum))
end

_init_data_deps() = map(_register_data, values(REMOTE_EXAMPLE_DATA))

@doc forwarddoc(:load_arviz_data) load_arviz_data

Base.@deprecate load_arviz_data(name; kwargs...) load_example_data(name; kwargs...)
Base.@deprecate load_arviz_data() load_example_data()
function load_arviz_data(name, data_home; kwargs...)
    Base.depwarn("customizing `data_home` is deprecated.", :data_home)
    return arviz.load_arviz_data(name, data_home; kwargs...)
end

Docs.getdoc(::typeof(load_arviz_data)) = forwardgetdoc(:load_arviz_data)
