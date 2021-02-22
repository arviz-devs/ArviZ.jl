@forwardfun bfmi
@forwardfun ess
@forwardfun mcse
@forwardfun rhat

function convert_arguments(::typeof(bfmi), data, args...; kwargs...)
    dataset = convert_to_dataset(data; group=:sample_stats)
    return tuple(dataset, args...), kwargs
end
function convert_arguments(::typeof(bfmi), data::AbstractArray, args...; kwargs...)
    return tuple(data, args...), kwargs
end

for f in (:ess, :mcse, :rhat)
    @eval begin
        function convert_arguments(::typeof($(f)), data, args...; kwargs...)
            dataset = convert_to_dataset(data; group=:posterior)
            return tuple(dataset, args...), kwargs
        end
        function convert_arguments(::typeof($(f)), data::AbstractArray, args...; kwargs...)
            return tuple(data, args...), kwargs
        end
    end
end
