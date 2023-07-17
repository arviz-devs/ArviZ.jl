"""
    compare(idata_pairs; elpd_method=loo, kwargs...)
    compare(elpd_pairs; kwargs...)

Compare models based on their expected log pointwise predictive density (ELPD).

The argument may be any object with a `pairs` method where each value is either an
[`InferenceData`](@ref) or an [`AbstractELPDResult`](@ref).

The ELPD is estimated either by Pareto smoothed importance sampling leave-one-out
cross-validation (LOO) or using the widely applicable information criterion (WAIC).
We recommend loo. Read more theory here - in a paper by some of the
leading authorities on model comparison dx.doi.org/10.1111/1467-9868.00353

# Keywords

  - `elpd_method=loo`: the method to be used to estimate ELPD from the data.
  - `method=:stacking`: the method to be used to combine ELPD estimates from different
    datasets. Currently only `:stacking` is supported.
"""
function compare(idata_or_elpd_pairs; elpd_method=loo, method::Symbol=:stacking)
    elpd_pairs = Dict(
        Iterators.map(pairs(idata_or_elpd_pairs)) do (name, idata_or_elpd)
            return Symbol(name) => _maybe_elpd(elpd_method, idata_or_elpd)
        end,
    )
    if method === :stacking
        return _compare_stacking(elpd_pairs)
    else
        throw(ArgumentError("Unsupported method: $method"))
    end
end

_maybe_elpd(elpd_method, data::InferenceData) = elpd_method(data)
_maybe_elpd(elpd_method, result::AbstractELPDResult) = result

_sphere_to_simplex(x) = x .^ 2
_simplex_to_sphere(x) = sqrt.(x)
function _∇sphere_to_simplex!(∂x, x)
    ∂x .*= 2 .* x
    return ∂x
end

function _compare_stacking(elpd_pairs)
    ic_mat = elpd_matrix(elpd_pairs)
    exp_ic_mat = exp.(ic_mat)
    z = similar(exp_ic_mat, axes(exp_ic_mat, 1))

    # if x ∈ 𝕊ⁿ, then x.^2 ∈ Δⁿ. This transformation is bijective if x is further
    # constrained to the positive orthant of the sphere, but it is sufficient to solve the
    # problem on any orthant.

    # objective function and gradient, avoiding redundant computation and allocations
    function fg!(F, G, x)
        w = _sphere_to_simplex(x)
        mul!(z, exp_ic_mat, w)
        z .= inv.(z)
        if G !== nothing
            mul!(G, exp_ic_mat', z)
            _∇sphere_to_simplex!(G, x)
            G .*= -1
        end
        if F !== nothing
            return sum(log, z)
        end
        return nothing
    end

    w0 = similar(exp_ic_mat, axes(exp_ic_mat, 2))
    w0 .= 1//length(w0)
    x0 = _simplex_to_sphere(w0)
    manifold = Optim.Sphere()
    sol = Optim.optimize(Optim.only_fg!(fg!), x0, Optim.LBFGS(; manifold))

    Optim.converged(sol) ||
        @warn "Optimization of stacking weights failed to converge after $(Optim.iterations(sol)) iterations."
    return collect(keys(elpd_pairs)), _sphere_to_simplex(sol.minimizer)
end

function elpd_matrix(elpd_results)
    elpd_values = Iterators.map(values(elpd_results)) do result
        return vec(elpd_estimates(result; pointwise=true).elpd)
    end
    elpd_mat = reduce(hcat, elpd_values)
    return elpd_mat
end
