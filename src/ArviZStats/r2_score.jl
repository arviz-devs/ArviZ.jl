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
idata = load_example_data("regression1d")
y_true = idata.observed_data.y
y_pred = PermutedDimsArray(idata.posterior_predictive.y, (:draw, :chain, :y_dim_0))
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
    r2_score(idata::InferenceData; y_name, y_pred_name) -> (; r2, r2_std)

Compute ``R²`` from `idata`, automatically formatting the predictions to the correct shape.

# Keywords

  - `y_name`: Name of observed data variable in `idata.observed_data`. If not provided, then
    the only observed data variable is used.
  - `y_pred_name`: Name of posterior predictive variable in `idata.posterior_predictive`.
    If not provided, then `y_name` is used.

# Examples

```jldoctest
using ArviZ, ArviZExampleData
idata = load_arviz_data("regression1d")
r2_score(idata)

# output

(r2 = 0.683196996216511, r2_std = 0.036883777654323734)
```
"""
function r2_score(
    idata::InferenceObjects.InferenceData;
    y_name::Union{Symbol,Nothing}=nothing,
    y_pred_name::Union{Symbol,Nothing}=nothing,
)
    (_, y), (_, y_pred) = observations_and_predictions(idata, y_name, y_pred_name)
    return r2_score(y, _draw_chains_params_array(y_pred))
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
    var_residual = dropdims(Statistics.var(y_pred .- y_true_reshape; corrected, dims); dims)

    # allocate storage for type-stability
    T = typeof(first(var_y_est) / first(var_residual))
    sample_axes = ntuple(Base.Fix1(axes, y_pred), ndims(y_pred) - 1)
    r_squared = similar(y_pred, T, sample_axes)
    r_squared .= var_y_est ./ (var_y_est .+ var_residual)
    return r_squared
end
