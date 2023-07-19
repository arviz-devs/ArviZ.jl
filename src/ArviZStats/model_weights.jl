"""
$(TYPEDEF)

An abstract type representing methods for computing model weights.

Subtypes implement [`model_weights`](@ref)`(method, elpd_results)`.
"""
abstract type AbstractModelWeightsMethod end

"""
    model_weights(method::AbstractModelWeightsMethod, elpd_results)

Compute weights for each model in `elpd_results` using `method`.

`elpd_results` is any iterator with `keys` and `values` methods, where each value is an
[`AbstractELPDResult`](@ref).

See also: [`AbstractModelWeightsMethod`](@ref), [`compare`](@ref)
"""
function model_weights end

"""
    model_weights(elpd_results)

Compute weights for each model in `elpd_results` using [`Stacking`](@ref).
"""
model_weights(elpd_results) = model_weights(Stacking(), elpd_results)

"""
$(TYPEDEF)

Model weighting method using pseudo Bayesian Model Averaging (pseudo-BMA) and Akaike-type
weighting.

    PseudoBMA(; regularized=true)

Construct the method with optional regularization of the weights using the standard error of
the ELPD estimate.

!!! note

    It is recommended to instead use [`BootstrappedPseudoBMA`](@ref), which produces more
    stable estimates of model weights. For details, see [^YaoVehtari2018].

[^YaoVehtari2018]: Yuling Yao, Aki Vehtari, Daniel Simpson, and Andrew Gelman.
    Using Stacking to Average Bayesian Predictive Distributions.
    2018. Bayesian Analysis. 13, 3, 917–1007.
    doi: [10.1214/17-BA1091](https://doi.org/10.1214/17-BA1091)
    arXiv: [1704.02030](https://arxiv.org/abs/1704.02030)
See also: [`Stacking`](@ref)
"""
@kwdef struct PseudoBMA <: AbstractModelWeightsMethod
    regularized::Bool = true
end

function model_weights(method::PseudoBMA, elpd_results)
    return LogExpFunctions.softmax(collect(
        Iterators.map(values(elpd_results)) do result
            est = elpd_estimates(result)
            method.regularized || return est.elpd
            return est.elpd - est.elpd_mcse / 2
        end,
    ))
end

"""
$(TYPEDEF)

Model weighting method using pseudo Bayesian Model Averaging using Akaike-type weighting
with the Bayesian bootstrap (pseudo-BMA+)[^YaoVehtari2018].

The Bayesian bootstrap stabilizes the model weights.

    BootstrappedPseudoBMA(; rng=Random.default_rng(), samples=1_000, alpha=1)

Construct the method.

$(TYPEDFIELDS)

See also: [`Stacking`](@ref)
"""
@kwdef struct BootstrappedPseudoBMA{R<:Random.AbstractRNG,T<:Real} <:
              AbstractModelWeightsMethod
    "The random number generator to use for the Bayesian bootstrap"
    rng::R = Random.default_rng()
    "The number of samples to draw for bootstrapping"
    samples::Int = 1_000
    """The shape parameter in the Dirichlet distribution used for the Bayesian bootstrap.
    `alpha < 1` favors more uniform weights."""
    alpha::T = 1
end

function model_weights(method::BootstrappedPseudoBMA, elpd_results)
    _elpd = elpd_estimates(first(values(elpd_results)); pointwise=true).elpd
    α = similar(_elpd)
    n = length(α)
    rng = method.rng
    α_dist = Distributions.Dirichlet(n, method.alpha)
    ic_mat = _elpd_matrix(elpd_results)
    elpd_mean = similar(ic_mat, axes(ic_mat, 2))
    weights_mean = zero(elpd_mean)
    w = similar(weights_mean)
    for _ in 1:(method.samples)
        _model_weights_bootstrap!(w, elpd_mean, α, rng, α_dist, ic_mat)
        weights_mean .+= w
    end
    weights_mean ./= method.samples
    return weights_mean
end

function _model_weights_bootstrap!(w, elpd_mean, α, rng, α_dist, ic_mat)
    Random.rand!(rng, α_dist, α)
    mul!(elpd_mean, ic_mat', α)
    elpd_mean .*= length(α)
    LogExpFunctions.softmax!(w, elpd_mean)
    return w
end

"""
$(TYPEDEF)

Model weighting using stacking of predictive distributions[^YaoVehtari2018].

    Stacking()

Construct the method.

See also: [`BootstrappedPseudoBMA`](@ref)

[^YaoVehtari2018]: Yuling Yao, Aki Vehtari, Daniel Simpson, and Andrew Gelman.
    Using Stacking to Average Bayesian Predictive Distributions.
    2018. Bayesian Analysis. 13, 3, 917–1007.
    doi: [10.1214/17-BA1091](https://doi.org/10.1214/17-BA1091)
    arXiv: [1704.02030](https://arxiv.org/abs/1704.02030)
"""
struct Stacking <: AbstractModelWeightsMethod end

function model_weights(::Stacking, elpd_pairs)
    ic_mat = _elpd_matrix(elpd_pairs)
    exp_ic_mat = exp.(ic_mat)
    _, weights = _model_weights_stacking(exp_ic_mat)
    return weights
end
function _model_weights_stacking(exp_ic_mat)
    # set up optimization objective
    manifold = Optim.Sphere()
    objective = InplaceStackingOptimObjective(manifold, exp_ic_mat)

    # set up initial point on optimization manifold
    w0 = similar(exp_ic_mat, axes(exp_ic_mat, 2))
    w0 .= 1//length(w0)
    x0 = _initial_point(objective, w0)

    # optimize
    sol = Optim.optimize(Optim.only_fg!(objective), x0, Optim.LBFGS(; manifold))

    # check convergence
    Optim.converged(sol) ||
        @warn "Optimization of stacking weights failed to converge after $(Optim.iterations(sol)) iterations."

    # return solution and weights
    w = _final_point(objective, sol.minimizer)
    return sol, w
end

function _elpd_matrix(elpd_results)
    elpd_values = Iterators.map(values(elpd_results)) do result
        return vec(elpd_estimates(result; pointwise=true).elpd)
    end
    elpd_mat = reduce(hcat, elpd_values)
    return elpd_mat
end

# if x ∈ 𝕊ⁿ, then x.^2 ∈ Δⁿ. This transformation is bijective if x is further
# constrained to the positive orthant of the sphere, but it is sufficient to solve the
# problem on any orthant.
struct InplaceStackingOptimObjective{M<:Optim.Manifold,E,C}
    manifold::M
    exp_ic_mat::E
    cache::C
end
function InplaceStackingOptimObjective(manifold, exp_ic_mat)
    return InplaceStackingOptimObjective(manifold, exp_ic_mat, nothing)
end

function InplaceStackingOptimObjective(manifold::Optim.Sphere, exp_ic_mat)
    cache = similar(exp_ic_mat, axes(exp_ic_mat, 1))
    return InplaceStackingOptimObjective(manifold, exp_ic_mat, cache)
end
function (obj::InplaceStackingOptimObjective{Optim.Sphere})(F, G, x)
    exp_ic_mat = obj.exp_ic_mat
    cache = obj.cache
    w = _sphere_to_simplex(x)
    mul!(cache, exp_ic_mat, w)
    cache .= inv.(cache)
    if G !== nothing
        mul!(G, exp_ic_mat', cache)
        G .*= -1
        _∇sphere_to_simplex!(G, x)
    end
    if F !== nothing
        return sum(log, cache)
    end
    return nothing
end
_initial_point(::InplaceStackingOptimObjective{Optim.Sphere}, w0) = _simplex_to_sphere(w0)
_final_point(::InplaceStackingOptimObjective{Optim.Sphere}, x) = _sphere_to_simplex(x)

_sphere_to_simplex(x) = x .^ 2
_simplex_to_sphere(x) = sqrt.(x)
function _∇sphere_to_simplex!(∂x, x)
    ∂x .*= 2 .* x
    return ∂x
end
