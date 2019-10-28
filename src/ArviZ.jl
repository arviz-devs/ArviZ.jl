__precompile__()
module ArviZ

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

using Reexport
using PyCall
@reexport using PyPlot

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

@delegate plot_autocorr plot_compare plot_density plot_dist plot_elpd plot_energy plot_ess plot_forest plot_hpd plot_joint plot_kde plot_khat plot_loo_pit plot_mcse plot_pair plot_parallel plot_posterior plot_ppc plot_rank plot_trace plot_violin

end # module
