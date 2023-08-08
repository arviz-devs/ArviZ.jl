module ArviZ

using Reexport

## Re-export component packages
@reexport using InferenceObjects
# only export recommended functions and corresponding utilities
@reexport using MCMCDiagnosticTools:
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
@reexport using PosteriorStats
@reexport using PSIS
@reexport using StatsBase: summarystats

## Conversions
export from_mcmcchains, from_samplechains

const EXTENSIONS_SUPPORTED = isdefined(Base, :get_extension)

include("utils.jl")
include("conversions.jl")

if !EXTENSIONS_SUPPORTED
    using Requires: @require
end
@static if !EXTENSIONS_SUPPORTED
    function __init__()
        @require SampleChains = "754583d1-7fc4-4dab-93b5-5eaca5c9622e" begin
            include("../ext/ArviZSampleChainsExt.jl")
        end
        @require SampleChainsDynamicHMC = "6d9fd711-e8b2-4778-9c70-c1dfb499d4c4" begin
            include("../ext/ArviZSampleChainsDynamicHMCExt.jl")
        end
        @require MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d" begin
            include("../ext/ArviZMCMCChainsExt.jl")
        end
    end
end

end # module
