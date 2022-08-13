function package_version(pkg::Module)
    isdefined(Base, :pkgversion) && return Base.pkgversion(pkg)
    project = joinpath(dirname(dirname(pathof(pkg))), "Project.toml")
    toml = read(project, String)
    m = match(r"(*ANYCRLF)^version\s*=\s\"(.*)\"$"m, toml)
    return VersionNumber(m[1])
end
