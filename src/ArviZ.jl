module ArviZ

using Base: @__doc__
using Requires
using REPL
using OrderedCollections: OrderedDict
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

using InferenceObjects
import InferenceObjects: convert_to_inference_data, namedtuple_of_arrays
# internal functions temporarily used/extended here
using InferenceObjects:
    attributes, recursive_stack, groupnames, groups, hasgroup, setattribute!
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

## Conversions
export from_mcmcchains, from_samplechains

const DEFAULT_SAMPLE_DIMS = Dimensions.key2dim((:chain, :draw))

include("utils.jl")
include("ArviZStats/ArviZStats.jl")
using .ArviZStats

include("conversions.jl")
@static if !isdefined(Base, :isdefined)
    function __init__()
        @require SampleChains = "754583d1-7fc4-4dab-93b5-5eaca5c9622e" begin
            include("../ext/ArviZSampleChainsExt.jl")
        end
        @require MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d" begin
            include("../ext/ArviZMCMCChainsExt.jl")
        end
    end
end

end # module
