"""
    flatten(x)

If `x` is an array of arrays, flatten into a single array whose dimensions are ordered with
dimensions of the outermost container first and innermost container last.
"""
flatten(x) = x
flatten(x::AbstractArray{<:Number}) = convert(Array, x)
function flatten(x::AbstractArray{S}) where {T<:Number,N,S<:AbstractArray{T,N}}
    ret = Array{T}(undef, (size(x)..., size(x[1])...))
    for k in keys(x)
        setindex!(ret, x[k], k, (Colon() for _ in 1:N)...)
    end
    return ret
end

function package_version(pkg::Module)
    isdefined(Base, :pkgversion) && return Base.pkgversion(pkg)
    project = joinpath(dirname(dirname(pathof(pkg))), "Project.toml")
    toml = read(project, String)
    m = match(r"(*ANYCRLF)^version\s*=\s\"(.*)\"$"m, toml)
    return VersionNumber(m[1])
end

rekey(d, keymap) = Dict(get(keymap, k, k) => d[k] for k in keys(d))
function rekey(d::NamedTuple, keymap)
    new_keys = map(k -> get(keymap, k, k), keys(d))
    return NamedTuple{new_keys}(values(d))
end
