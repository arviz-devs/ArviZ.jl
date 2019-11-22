"""
    stack(x::NamedTuple) -> NamedTuple
    stack(x::AbstractArray{NamedTuple}) -> NamedTuple
    stack(x::AbstractArray{AbstractArray{<:NamedTuple}}) -> NamedTuple

Given a container of `NamedTuple`s, concatenate them, using the container
dimensions as the dimensions of the resulting arrays.

# Examples

```@example
using ArviZ
nchains, ndraws = 4, 100
data = [(x = rand(), y = randn(2), z = randn(2, 3)) for _ in 1:nchains, _ in 1:ndraws];
stacked_data = ArviZ.stack(data);
```
"""
stack(x::AbstractArray) = stack(stack.(x))
stack(x::AbstractArray{T}) where {T<:Number} = x
stack(x::NamedTuple) = x

function stack(x::AbstractArray{S}) where {T<:Number,N,S<:AbstractArray{T,N}}
    ret = Array{T}(undef, (size(x)..., size(x[1])...))
    @simd for k in keys(x)
        @inbounds setindex!(ret, x[k], k, (Colon() for _ = 1:N)...)
    end
    return ret
end

function stack(x::AbstractArray{<:NamedTuple{K}}) where {K}
    length(x) == 0 && return
    @inbounds x1 = x[1]
    ret = NamedTuple()
    for k in K
        v = replacemissing.(getproperty.(x, k))
        ret = merge(ret, (k => stack(v),))
    end
    return ret
end

