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

const LOCAL_DATA_DIR = joinpath(pkgdir(ArviZ), "deps", "data")

const EXAMPLE_DATA = Dict(
    "centered_eight" => LocalFileMetadata(;
        name="centered_eight",
        filename="centered_eight.nc",
        path=joinpath(LOCAL_DATA_DIR, "centered_eight"),
        description="""
A centered parameterization of the eight schools model.
Provided as an example of a model that NUTS has trouble fitting.
Compare to `non_centered_eight`.

The eight schools model is a hierarchical model used for an analysis of the effectiveness
of classes that were designed to improve students' performance on the Scholastic Aptitude
Test.

See Bayesian Data Analysis (Gelman et. al.) for more details.""",
    ),
    "non_centered_eight" => LocalFileMetadata(;
        name="non_centered_eight",
        filename="non_centered_eight.nc",
        path=joinpath(LOCAL_DATA_DIR, "non_centered_eight"),
        description="""
A non-centered parameterization of the eight schools model.
This is a hierarchical model where sampling problems may be fixed by a non-centered
parametrization.
Compare to `centered_eight`.

The eight schools model is a hierarchical model used for an analysis of the effectiveness of
classes that were designed to improve students' performance on the Scholastic Aptitude Test.

See Bayesian Data Analysis (Gelman et. al.) for more details.""",
    ),
    "radon" => RemoteFileMetadata(;
        name="radon",
        filename="radon_hierarchical.nc",
        url="http://figshare.com/ndownloader/files/24067472",
        checksum="a9b2b4adf1bf9c5728e5bdc97107e69c4fc8d8b7d213e9147233b57be8b4587b",
        description="""
Radon is a radioactive gas that enters homes through contact points with the ground.
It is a carcinogen that is the primary cause of lung cancer in non-smokers. Radon
levels vary greatly from household to household.

This example uses an EPA study of radon levels in houses in Minnesota to construct a
model with a hierarchy over households within a county. The model includes estimates
(gamma) for contextual effects of the uranium per household.

See Gelman and Hill (2006) for details on the example, or
https://docs.pymc.io/notebooks/multilevel_modeling.html
by Chris Fonnesbeck for details on this implementation.""",
    ),
    "rugby" => RemoteFileMetadata(;
        name="rugby",
        filename="rugby.nc",
        url="http://figshare.com/ndownloader/files/16254359",
        checksum="9eecd2c6317e45b0388dd97ae6326adecf94128b5a7d15a52c9fcfac0937e2a6",
        description="""
The Six Nations Championship is a yearly rugby competition between Italy, Ireland,
Scotland, England, France and Wales. Fifteen games are played each year, representing
all combinations of the six teams.

This example uses and includes results from 2014 - 2017, comprising 60 total
games. It models latent parameters for each team's attack and defense, as well
as a parameter for home team advantage.

See https://docs.pymc.io/notebooks/rugby_analytics.html by Peader Coyle
for more details and references.""",
    ),
    "regression1d" => RemoteFileMetadata(;
        name="regression1d",
        filename="regression1d.nc",
        url="http://figshare.com/ndownloader/files/16254899",
        checksum="909e8ffe344e196dad2730b1542881ab5729cb0977dd20ba645a532ffa427278",
        description="""
A synthetic one dimensional linear regression dataset with latent slope,
intercept, and noise ("eps"). One hundred data points, fit with PyMC3.

True slope and intercept are included as deterministic variables.""",
    ),
    "regression10d" => RemoteFileMetadata(;
        name="regression10d",
        filename="regression10d.nc",
        url="http://figshare.com/ndownloader/files/16255736",
        checksum="c6716ec7e19926ad2a52d6ae4c1d1dd5ddb747e204c0d811757c8e93fcf9f970",
        description="""
A synthetic multi-dimensional (10 dimensions) linear regression dataset with
latent weights ("w"), intercept, and noise ("eps"). Five hundred data points,
fit with PyMC3.

True weights and intercept are included as deterministic variables.""",
    ),
    "classification1d" => RemoteFileMetadata(;
        name="classification1d",
        filename="classification1d.nc",
        url="http://figshare.com/ndownloader/files/16256678",
        checksum="1cf3806e72c14001f6864bb69d89747dcc09dd55bcbca50aba04e9939daee5a0",
        description="""
A synthetic one dimensional logistic regression dataset with latent slope and
intercept, passed into a Bernoulli random variable. One hundred data points,
fit with PyMC3.

True slope and intercept are included as deterministic variables.""",
    ),
    "classification10d" => RemoteFileMetadata(;
        name="classification10d",
        filename="classification10d.nc",
        url="http://figshare.com/ndownloader/files/16256681",
        checksum="16c9a45e1e6e0519d573cafc4d266d761ba347e62b6f6a79030aaa8e2fde1367",
        description="""
A synthetic multi dimensional (10 dimensions) logistic regression dataset with
latent weights ("w") and intercept, passed into a Bernoulli random variable.
Five hundred data points, fit with PyMC3.

True weights and intercept are included as deterministic variables.""",
    ),
    "glycan_torsion_angles" => RemoteFileMetadata(;
        name="glycan_torsion_angles",
        filename="glycan_torsion_angles.nc",
        url="http://figshare.com/ndownloader/files/22882652",
        checksum="4622621fe7a1d3075c18c4c34af8cc57c59eabbb3501b20c6e2d9c6c4737034c",
        description="""
Torsion angles phi and psi are critical for determining the three dimensional
structure of bio-molecules. Combinations of phi and psi torsion angles that
produce clashes between atoms in the bio-molecule result in high energy, unlikely structures.

This model uses a Von Mises distribution to propose torsion angles for the
structure of a glycan molecule (pdb id: 2LIQ), and a Potential to estimate
the proposed structure's energy. Said Potential is bound by Boltzman's law.""",
    ),
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
    metadata = get(EXAMPLE_DATA, name, nothing)
    metadata === nothing && throw(
        ArgumentError("Dataset $name not found. Valid datasets are: $(keys(EXAMPLE_DATA))"),
    )
    filename = joinpath(metadata.name, metadata.filename)
    # none of our downloads require license agreements, so we always agree to the download
    path = withenv("DATADEPS_ALWAYS_ACCEPT" => true) do
        return DataDeps.resolve(filename, @__FILE__)
    end
    return from_netcdf(path; kwargs...)
end
load_example_data() = EXAMPLE_DATA

function _register_data(md::LocalFileMetadata)
    return DataDeps.register(DataDeps.ManualDataDep(md.name, md.description))
end

function _register_data(md::RemoteFileMetadata)
    return DataDeps.register(DataDeps.DataDep(md.name, md.description, md.url, md.checksum))
end

_init_data_deps() = map(_register_data, values(EXAMPLE_DATA))

@doc forwarddoc(:load_arviz_data) load_arviz_data

Base.@deprecate load_arviz_data(name; kwargs...) load_example_data(name; kwargs...)
Base.@deprecate load_arviz_data() load_example_data()
function load_arviz_data(name, data_home; kwargs...)
    Base.depwarn("customizing `data_home` is deprecated.", :data_home)
    return arviz.load_arviz_data(name, data_home; kwargs...)
end

Docs.getdoc(::typeof(load_arviz_data)) = forwardgetdoc(:load_arviz_data)
