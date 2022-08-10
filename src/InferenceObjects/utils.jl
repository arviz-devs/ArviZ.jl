function package_version(pkg::Module)
    isdefined(Base, :pkgversion) && return Base.pkgversion(pkg)
    project = joinpath(dirname(dirname(pathof(pkg))), "Project.toml")
    toml = read(project, String)
    m = match(r"(*ANYCRLF)^version\s*=\s\"(.*)\"$"m, toml)
    return VersionNumber(m[1])
end

struct AsSlice{T<:Dimensions.Selector} <: Dimensions.Selector{T}
    val::T
end

function Dimensions.selectindices(l::Dimensions.LookupArray, sel::AsSlice; kw...)
    i = Dimensions.selectindices(l, Dimensions.val(sel); kw...)
    inds = i isa AbstractVector ? i : [i]
    return inds
end
