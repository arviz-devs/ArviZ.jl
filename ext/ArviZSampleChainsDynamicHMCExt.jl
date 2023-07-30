module ArviZSampleChainsDynamicHMCExt

using ..SampleChains: SampleChains
using ..SampleChainsDynamicHMC: SampleChainsDynamicHMC

function _samplechains_info(chain::SampleChainsDynamicHMC.DynamicHMCChain)
    info = SampleChains.info(chain)
    termination = info.termination
    tree_stats = (
        energy=info.π,
        tree_depth=info.depth,
        acceptance_rate=info.acceptance_rate,
        n_steps=info.steps,
        diverging=map(t -> t.left == t.right, termination),
        turning=map(t -> t.left < t.right, termination),
    )
    used_info = (:π, :depth, :acceptance_rate, :steps, :termination)
    skipped_info = setdiff(propertynames(info), used_info)
    isempty(skipped_info) ||
        @debug "Skipped SampleChainsDynamicHMC info entries: $skipped_info."
    return tree_stats
end

end  # module
