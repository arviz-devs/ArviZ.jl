__precompile__()
module ArviZ

using Base: @__doc__
using Requires
using REPL
using NamedTupleTools
using DataFrames

using PyCall
if PyCall.conda
    using Conda
    Conda.add_channel("conda-forge") # try to avoid mixing channels
end
using PyPlot

import Base:
    convert,
    get,
    getproperty,
    hash,
    haskey,
    iterate,
    length,
    propertynames,
    setindex,
    show,
    write,
    +
import Base.Docs: getdoc
import StatsBase
import StatsBase: summarystats
import Markdown: @doc_str
import PyCall: PyObject

# Exports

## Plots
export plot_autocorr,
    plot_compare,
    plot_density,
    plot_dist,
    plot_elpd,
    plot_energy,
    plot_ess,
    plot_forest,
    plot_hpd,
    plot_joint,
    plot_kde,
    plot_khat,
    plot_loo_pit,
    plot_mcse,
    plot_pair,
    plot_parallel,
    plot_posterior,
    plot_ppc,
    plot_rank,
    plot_trace,
    plot_violin

## Stats
export summarystats, compare, hpd, loo, loo_pit, psislw, r2_score, waic

## Diagnostics
export bfmi, geweke, ess, rhat, mcse

## Stats utils
export autocov, autocorr, make_ufunc, wrap_xarray_ufunc

## Data
export InferenceData,
    convert_to_inference_data,
    load_arviz_data,
    to_netcdf,
    from_netcdf,
    from_namedtuple,
    from_dict,
    from_cmdstan,
    from_mcmcchains,
    concat,
    concat!

## Utils
export with_interactive_backend

## rcParams
export rcParams, with_rc_context

import_arviz() = pyimport_conda("arviz", "arviz", "conda-forge")

const arviz = import_arviz() # Load ArviZ once at precompilation time for docstrings
const xarray = PyNULL()
const bokeh = PyNULL()
const selenium = PyNULL()
const pandas = PyNULL()
const _min_arviz_version = v"0.6.1"
const _rcParams = PyNULL()

arviz_version() = VersionNumber(arviz.__version__)

const _precompile_arviz_version = arviz_version()

function initialize_arviz()
    ispynull(arviz) || return
    copy!(arviz, import_arviz())
    if arviz_version() != _precompile_arviz_version
        @warn "ArviZ.jl was precompiled using arviz version $(_precompile_version) but loaded with version $(arviz_version()). Please recompile with `using Pkg; Pkg.build('ArviZ')`."
    end
    if arviz_version() < _min_arviz_version
        @warn "ArviZ.jl only officially supports arviz version $(_min_arviz_version) or greater but found version $(arviz_version()). Please update."
    end

    pytype_mapping(arviz.InferenceData, InferenceData)

    # pytypemap-ing RcParams produces a Dict
    copy!(_rcParams, py"$(arviz).rcparams.rcParams"o)

    # use 1-based indexing by default within arviz
    rcParams["data.index_origin"] = 1

    # handle Bokeh showing ourselves
    rcParams["plot.bokeh.show"] = false

    initialize_xarray()
    initialize_numpy()
end

function initialize_xarray()
    ispynull(xarray) || return
    copy!(xarray, pyimport_conda("xarray", "xarray", "conda-forge"))
    pyimport_conda("dask", "dask", "conda-forge")
    pytype_mapping(xarray.Dataset, Dataset)
end

function initialize_numpy()
    # Trigger NumPy initialization, see https://github.com/JuliaPy/PyCall.jl/issues/744
    PyObject([true])
end

function initialize_pandas()
    ispynull(pandas) || return
    copy!(pandas, pyimport_conda("pandas", "pandas", "conda-forge"))
end

function __init__()
    initialize_arviz()
    @require MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca" begin
        import .MonteCarloMeasurements: AbstractParticles
        include("particles.jl")
    end
    @require MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d" begin
        import .MCMCChains: Chains, sections
        include("mcmcchains.jl")
    end
end

include("utils.jl")
include("rcparams.jl")
include("dataset.jl")
include("data.jl")
include("diagnostics.jl")
include("plots.jl")
include("bokeh.jl")
include("stats.jl")
include("stats_utils.jl")
include("namedtuple.jl")

end # module
