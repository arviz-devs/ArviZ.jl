module ArviZStats

using ArviZ: ArviZ, arviz, @forwardfun
using DataInterpolations: DataInterpolations
using DimensionalData: DimensionalData, Dimensions
using Distributions: Distributions
using DocStringExtensions: FIELDS, FUNCTIONNAME, TYPEDEF, TYPEDFIELDS, SIGNATURES
using InferenceObjects: InferenceObjects
using IteratorInterfaceExtensions: IteratorInterfaceExtensions
using LinearAlgebra: mul!, norm
using LogExpFunctions: LogExpFunctions
using Markdown: @doc_str
using MCMCDiagnosticTools: MCMCDiagnosticTools
using Optim: Optim
using PrettyTables: PrettyTables
using Printf: Printf
using PSIS: PSIS, PSISResult, psis, psis!
using PyCall: PyCall
using Random: Random
using Setfield: Setfield
using Statistics: Statistics
using StatsBase: StatsBase
using Tables: Tables
using TableTraits: TableTraits

# PSIS
export PSIS, PSISResult, psis, psis!

# LOO-CV
export AbstractELPDResult, PSISLOOResult, WAICResult
export elpd_estimates, information_criterion, loo, waic

# Model weighting and comparison
export AbstractModelWeightsMethod, BootstrappedPseudoBMA, PseudoBMA, Stacking, model_weights
export ModelComparisonResult, compare

# Others
export hdi, hdi!, loo_pit, r2_score

# load for docstrings
using ArviZ: InferenceData, convert_to_dataset, ess

const INFORMATION_CRITERION_SCALES = (deviance=-2, log=1, negative_log=-1)

include("utils.jl")
include("hdi.jl")
include("elpdresult.jl")
include("loo.jl")
include("waic.jl")
include("model_weights.jl")
include("compare.jl")
include("loo_pit.jl")
include("r2_score.jl")

end  # module
