__precompile__()
module ArviZ

using Reexport
using PyCall
@reexport using PyPlot
using Pandas: DataFrame
using MCMCChains: AbstractChains

import Base: getproperty, show, summary, +

export plot_autocorr,
       plot_density,
       plot_dist,
       plot_energy,
       plot_ess,
       plot_forest,
       plot_hpd,
       plot_joint,
       plot_kde,
       plot_loo_pit,
       plot_mcse,
       plot_pair,
       plot_parallel,
       plot_posterior,
       plot_ppc,
       plot_rank,
       plot_trace,
       plot_violin,
       InferenceData,
       convert_to_inference_data,
       from_dict,
       from_mcmcchains,
       concat

const arviz = PyNULL()

function __init__()
    copy!(arviz, pyimport_conda("arviz", "arviz"))
end

include("utils.jl")
include("data.jl")
include("stats.jl")
include("plots.jl")
include("mcmcchains.jl")

end # module
