__precompile__()
module ArviZ

using Base: @__doc__
using Requires
using REPL
using PyCall
using PyPlot
using Pandas
using NamedTupleTools

import Base: convert, propertynames, getproperty, hash, show, +
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
       concat,
       concat!

## Utils
export with_interactive_backend

## rcParams
export with_rc_context

# Load ArviZ once at precompilation time for docstrings
import_arviz() = pyimport_conda("arviz", "arviz", "conda-forge")

const arviz = import_arviz()
const xarray = PyNULL()
const bokeh = PyNULL()

function __init__()
    copy!(arviz, import_arviz())
    copy!(xarray, pyimport_conda("xarray", "xarray", "conda-forge"))
    pyimport_conda("dask", "dask", "conda-forge")

    try
        copy!(bokeh, pyimport_conda("bokeh", "bokeh", "conda-forge"))
    catch
    end

    if !ispynull(bokeh)
        pytype_mapping(pyimport("bokeh.model").Model, BokehPlot)
        pytype_mapping(pyimport("bokeh.document").Document, BokehPlot)
    end

    pytype_mapping(xarray.Dataset, Dataset)
    pytype_mapping(arviz.InferenceData, InferenceData)

    @require MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d" include("mcmcchains.jl")
end

include("utils.jl")
include("bokeh.jl")
include("dataset.jl")
include("data.jl")
include("diagnostics.jl")
include("plots.jl")
include("stats.jl")
include("stats_utils.jl")
include("namedtuple.jl")

end # module
