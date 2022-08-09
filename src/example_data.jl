abstract type AbstractFileMetadata end

Base.@kwdef struct LocalFileMetadata <: AbstractFileMetadata
    name::String
    filename::String
    path::String
    description::String = ""
end

function Base.show(io::IO, ::MIME"text/plain", md::LocalFileMetadata)
    println(io, md.name)
    println(io, repeat("=", length(md.name)))
    println(io)
    println(io, md.description)
    println(io)
    print(io, "local: ", joinpath(md.path, md.filename))
    return nothing
end

Base.@kwdef struct RemoteFileMetadata <: AbstractFileMetadata
    name::String
    filename::String
    url::String
    checksum::String = ""
    description::String = ""
end

function Base.show(io::IO, ::MIME"text/plain", md::RemoteFileMetadata)
    println(io, md.name)
    println(io, repeat("=", length(md.name)))
    println(io)
    println(io, md.description)
    println(io)
    print(io, "remote: ", md.url)
    return nothing
end

