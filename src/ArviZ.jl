__precompile__()
module ArviZ

using Base: @__doc__
using Requires
using REPL
using DataFrames
using OrderedCollections: OrderedDict

using PyCall
using Conda
using PyPlot
using DimensionalData: DimensionalData, Dimensions
using LogExpFunctions: logsumexp

import Base:
    convert,
    get,
    getindex,
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

using InferenceObjects
import InferenceObjects: convert_to_inference_data, namedtuple_of_arrays
# internal functions temporarily used/extended here
using InferenceObjects:
    attributes, recursive_stack, groupnames, groups, hasgroup, rekey, setattribute!
import InferenceObjects: namedtuple_of_arrays
using InferenceObjects: from_netcdf, to_netcdf

using MCMCDiagnosticTools:
    MCMCDiagnosticTools,
    AutocovMethod,
    FFTAutocovMethod,
    BDAAutocovMethod,
    bfmi,
    ess,
    ess_rhat,
    mcse,
    rhat,
    rstar

# Exports

## Stats
export ArviZStats
export AbstractELPDResult, PSISLOOResult, WAICResult
export PSIS, PSISResult, psis, psis!
export elpd_estimates, information_criterion, loo, waic
export AbstractModelWeightsMethod, BootstrappedPseudoBMA, PseudoBMA, Stacking, model_weights
export ModelComparisonResult, compare
export hdi, hdi!, loo_pit, r2_score

## Diagnostics
export MCMCDiagnosticTools, AutocovMethod, FFTAutocovMethod, BDAAutocovMethod
export bfmi, ess, ess_rhat, mcse, rhat, rstar

## InferenceObjects
export InferenceObjects,
    Dataset,
    InferenceData,
    convert_to_dataset,
    convert_to_inference_data,
    from_dict,
    from_namedtuple,
    namedtuple_to_dataset

## NetCDF I/O
export from_netcdf, to_netcdf

## Data
export from_mcmcchains, from_samplechains

## rcParams
export rcParams, with_rc_context

const _min_arviz_version = v"0.13.0"
const arviz = PyNULL()
const xarray = PyNULL()
const pandas = PyNULL()
const _rcParams = PyNULL()
const DEFAULT_SAMPLE_DIMS = Dimensions.key2dim((:chain, :draw))
const SUPPORTED_GROUPS = Symbol[]
const SUPPORTED_GROUPS_DICT = Dict{Symbol,Int}()

include("setup.jl")

# Load ArviZ once at precompilation time for docstringS
copy!(arviz, import_arviz())
check_needs_update(; update=false)
const _precompile_arviz_version = arviz_version()

function __init__()
    initialize_arviz()
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
include("xarray.jl")

include("ArviZStats/ArviZStats.jl")
using .ArviZStats
using .ArviZStats: summary


end # module
