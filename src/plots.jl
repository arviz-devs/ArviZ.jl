@forwardplotfun plot_autocorr
@forwardplotfun plot_bpv
@forwardplotfun plot_compare
@forwardplotfun plot_density
@forwardplotfun plot_dist
@forwardplotfun plot_dist_comparison
@forwardplotfun plot_elpd
@forwardplotfun plot_energy
@forwardplotfun plot_ess
@forwardplotfun plot_forest
@forwardplotfun plot_hdi
@forwardplotfun plot_kde
@forwardplotfun plot_khat
@forwardplotfun plot_loo_pit
@forwardplotfun plot_mcse
@forwardplotfun plot_pair
@forwardplotfun plot_parallel
@forwardplotfun plot_posterior
@forwardplotfun plot_ppc
@forwardplotfun plot_rank
@forwardplotfun plot_separation
@forwardplotfun plot_trace
@forwardplotfun plot_violin

@deprecate plot_joint(args...; kwargs...) plot_pair(args...; kwargs...)

function convert_arguments(::typeof(plot_compare), df, args...; kwargs...)
    pdf = topandas(Val(:DataFrame), df; index_name=:name)
    return tuple(pdf, args...), kwargs
end

function convert_arguments(::typeof(plot_elpd), data, args...; kwargs...)
    dict = Dict(k => try
        topandas(Val(:ELPDData), v)
    catch
        convert_to_inference_data(v)
    end for (k, v) in pairs(data))
    return tuple(dict, args...), kwargs
end
function convert_arguments(::typeof(plot_khat), df, args...; kwargs...)
    return tuple(topandas(Val(:ELPDData), df), args...), kwargs
end

for f in (
    :plot_autocorr,
    :plot_ess,
    :plot_mcse,
    :plot_pair,
    :plot_parallel,
    :plot_posterior,
    :plot_trace,
    :plot_violin,
)
    @eval begin
        function convert_arguments(::typeof($(f)), data, args...; kwargs...)
            idata = convert_to_inference_data(data; group=:posterior)
            return tuple(idata, args...), kwargs
        end
    end
end

for f in (:plot_autocorr, :plot_ess, :plot_mcse, :plot_posterior, :plot_violin)
    @eval begin
        function convert_arguments(::typeof($(f)), data::AbstractArray, args...; kwargs...)
            return tuple(data, args...), kwargs
        end
    end
end

function convert_arguments(::typeof(plot_energy), data, args...; kwargs...)
    dataset = convert_to_dataset(data; group=:sample_stats)
    return tuple(dataset, args...), kwargs
end

for f in (:plot_density, :plot_forest, :plot_rank)
    @eval begin
        function convert_arguments(
            ::typeof($(f)), data, args...; transform=identity, group=:posterior, kwargs...
        )
            tdata = transform(data)
            dataset = convert_to_dataset(tdata; group)
            return tuple(dataset, args...), kwargs
        end
    end
end

for f in (:plot_density, :plot_forest)
    @eval begin
        function convert_arguments(
            ::typeof($(f)),
            data::Union{AbstractVector,Tuple},
            transform=identity,
            group=:posterior,
            args...;
            kwargs...,
        )
            tdata = transform(data)
            datasets = map(tdata) do datum
                return convert_to_dataset(datum; group)
            end
            return tuple(datasets, args...), kwargs
        end
        function convert_arguments(
            ::typeof($(f)),
            data::AbstractVector{<:Real},
            transform=identity,
            group=:posterior,
            args...;
            kwargs...,
        )
            tdata = transform(data)
            dataset = convert_to_dataset(tdata; group)
            return tuple(dataset, args...), kwargs
        end
    end
end
