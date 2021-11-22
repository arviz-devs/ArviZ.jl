__precompile__()
module ArviZ

using Base: @__doc__
using Requires
using REPL
using NamedTupleTools
using DataFrames

using PyCall
using Conda
using PyPlot
using PSIS: PSISResult, psis, psis!
using LogExpFunctions: logsumexp

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
using StatsBase: StatsBase
import StatsBase: summarystats
import Markdown: @doc_str
import PyCall: PyObject

# Exports

## Plots
export plot_autocorr,
    plot_bpv,
    plot_compare,
    plot_density,
    plot_dist,
    plot_dist_comparison,
    plot_elpd,
    plot_energy,
    plot_ess,
    plot_forest,
    plot_hdi,
    plot_kde,
    plot_khat,
    plot_loo_pit,
    plot_mcse,
    plot_pair,
    plot_parallel,
    plot_posterior,
    plot_ppc,
    plot_rank,
    plot_separation,
    plot_trace,
    plot_violin

## Stats
export PSISResult, psis, psis!, psislw
export summarystats, compare, hdi, loo, loo_pit, r2_score, waic

## Diagnostics
export bfmi, ess, rhat, mcse

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
    from_samplechains,
    concat,
    concat!

## Utils
export with_interactive_backend

## rcParams
export rcParams, with_rc_context

const _min_arviz_version = v"0.11.3"
const arviz = PyNULL()
const xarray = PyNULL()
const bokeh = PyNULL()
const pandas = PyNULL()
const _rcParams = PyNULL()

include("setup.jl")

# Load ArviZ once at precompilation time for docstringS
copy!(arviz, import_arviz())
check_needs_update(; update=false)
const _precompile_arviz_version = arviz_version()

function __init__()
    initialize_arviz()
    @require MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca" begin
        import .MonteCarloMeasurements: AbstractParticles
        include("particles.jl")
    end
    @require SampleChains = "754583d1-7fc4-4dab-93b5-5eaca5c9622e" begin
        include("samplechains.jl")
    end
    @require MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d" begin
        import .MCMCChains: Chains, sections
        include("mcmcchains.jl")
    end
    return nothing
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
