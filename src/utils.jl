# Adapted from https://github.com/JuliaPy/Seaborn.jl
macro delegate(f_list...)
    blocks = Expr(:block)
    for f in f_list
        block = quote
            function $(esc(f))(args...; kwargs...)
                data = convert_to_arviz_data(first(args))
                arviz.$(f)(data, Base.tail(args)...; kwargs...)
            end
        end
        push!(blocks.args, block)
    end
    blocks
end
