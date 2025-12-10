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
export from_mcmcchains

const EXTENSIONS_SUPPORTED = isdefined(Base, :get_extension)

include("utils.jl")
include("conversions.jl")

if !EXTENSIONS_SUPPORTED
    using Requires: @require
end
@static if !EXTENSIONS_SUPPORTED
    function __init__()
        @require MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d" begin
            include("../ext/ArviZMCMCChainsExt.jl")
        end
    end
end

end # module
