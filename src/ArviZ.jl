__precompile__()
module ArviZ

using Base: @__doc__
using Reexport
using PyCall
@reexport using PyPlot
using Pandas: DataFrame, Series
using MCMCChains: AbstractChains

import Base: convert, propertynames, getproperty, hash, show, summary, +
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
export compare, hpd, loo, loo_pit, psislw, r2_score, waic

## Data
export InferenceData,
       convert_to_inference_data,
       load_arviz_data,
       to_netcdf,
       from_netcdf,
       from_dict,
       from_mcmcchains,
       concat

const arviz = PyNULL()

function __init__()
    copy!(arviz, pyimport_conda("arviz", "arviz"))

    pytype_mapping(arviz.InferenceData, InferenceData)
end

include("utils.jl")
include("data.jl")
include("stats.jl")
include("plots.jl")
include("mcmcchains.jl")

end # module
