"""
    r2_score(y_true::AbstractVector, y_pred::AbstractVecOrMat) -> (; r2, r2_std)

``R²`` for Bayesian regression models. Only valid for linear models.

# Arguments

  - `y_true`: Ground truth (correct) target values of length `noutputs`
  - `y_pred`: Estimated target values with size `(ndraws[, nchains], noutputs)`

See also [`r2_samples`](@ref).

# Example

```jldoctest
using ArviZ, ArviZExampleData
y_true = data.observed_data.y
y_pred = PermutedDimsArray(data.posterior_predictive.y, (:draw, :chain, :y_dim_0))
r2_score(y_true, y_pred)

# output

(r2 = 0.683196996216511, r2_std = 0.036883777654323734)
```
"""
function r2_score(y_true, y_pred)
    r_squared = r2_samples(y_true, y_pred)
    return NamedTuple{(:r2, :r2_std)}(StatsBase.mean_and_std(r_squared; corrected=false))
end

"""
    r2_samples(y_true::AbstractVector, y_pred::AbstractMatrix) -> AbstractVector

``R²`` samples for Bayesian regression models. Only valid for linear models.

See also [`r2_score`](@ref).

# Arguments

  - `y_true`: Ground truth (correct) target values of length `noutputs`
  - `y_pred`: Estimated target values with size `(ndraws[, nchains], noutputs)`
"""
function r2_samples(y_true::AbstractVector, y_pred::AbstractArray)
    @assert ndims(y_pred) ∈ (2, 3)
    corrected = false
    dims = output_dim = ndims(y_pred)
    sample_dims = ntuple(identity, ndims(y_pred) - 1)

    var_y_est = dropdims(Statistics.var(y_pred; corrected, dims); dims)
    y_true_reshape = reshape(y_true, ntuple(one, ndims(y_pred) - 1)..., :)
    var_e = dropdims(Statistics.var(y_pred .- y_true_reshape; corrected, dims); dims)

    # allocate storage for type-stability
    T = typeof(first(var_y_est) / first(var_e))
    sample_axes = ntuple(Base.Fix1(axes, y_pred), ndims(y_pred) - 1)
    r_squared = similar(y_pred, T, sample_axes)
    r_squared .= var_y_est ./ (var_y_est .+ var_e)
    return r_squared
end
