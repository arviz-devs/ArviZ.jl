__precompile__()
module ArviZ

using Reexport
using PyCall
@reexport using PyPlot

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
    plot_violin,
    convert_to_inference_data


const arviz = PyNULL()

function __init__()
    copy!(arviz, pyimport_conda("arviz", "arviz"))
end

# Adapted from https://github.com/JuliaPy/Seaborn.jl
macro delegate(f_list...)
    blocks = Expr(:block)
    for f in f_list
        block = quote
            function $(esc(f))(args...; kwargs...)
                arviz.$(f)(args...; kwargs...)
            end
        end
        push!(blocks.args, block)
    end
    blocks
end

@delegate plot_autocorr plot_compare plot_density plot_dist plot_elpd plot_energy plot_ess plot_forest plot_hpd plot_joint plot_kde plot_khat plot_loo_pit plot_mcse plot_pair plot_parallel plot_posterior plot_ppc plot_rank plot_trace plot_violin convert_to_inference_data

end # module
