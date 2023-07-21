const DEFAULT_STACKING_OPTIMIZER = Optim.LBFGS()

"""
$(TYPEDEF)

An abstract type representing methods for computing model weights.

Subtypes implement [`model_weights`](@ref)`(method, elpd_results)`.
"""
abstract type AbstractModelWeightsMethod end

"""
    model_weights(elpd_results; method=Stacking())
    model_weights(method::AbstractModelWeightsMethod, elpd_results)

Compute weights for each model in `elpd_results` using `method`.

`elpd_results` is a `Tuple`, `NamedTuple`, or `AbstractVector` with
[`AbstractELPDResult`](@ref) entries. The weights are returned in the same type of
collection.

[`Stacking`](@ref) is the recommended approach, as it performs well even when the true data
generating process is not included among the candidate models. See [^YaoVehtari2018] for
details.

See also: [`AbstractModelWeightsMethod`](@ref), [`compare`](@ref)

[^YaoVehtari2018]: Yuling Yao, Aki Vehtari, Daniel Simpson, and Andrew Gelman.
    Using Stacking to Average Bayesian Predictive Distributions.
    2018. Bayesian Analysis. 13, 3, 917–1007.
    doi: [10.1214/17-BA1091](https://doi.org/10.1214/17-BA1091)
    arXiv: [1704.02030](https://arxiv.org/abs/1704.02030)
# Examples

```jldoctest
using ArviZ
models = (
    centered=load_example_data("centered_eight"),
    non_centered=load_example_data("non_centered_eight"),
)
elpd_results = map(loo, models)
model_weights(elpd_results)

# output

(centered = 5.341753943391326e-19, non_centered = 1.0)
```
"""
function model_weights(elpd_results; method::AbstractModelWeightsMethod=Stacking())
    return model_weights(method, elpd_results)
end

# Akaike-type weights are defined as exp(-AIC/2), normalized to 1, which on the log-score
# IC scale is equivalent to softmax
akaike_weights!(w, elpds) = LogExpFunctions.softmax!(w, elpds)
_akaike_weights(elpds) = _softmax(elpds)

"""
$(TYPEDEF)

Model weighting method using pseudo Bayesian Model Averaging (pseudo-BMA) and Akaike-type
weighting.

    PseudoBMA(; regularize=false)
    PseudoBMA(regularize)

Construct the method with optional regularization of the weights using the standard error of
the ELPD estimate.

!!! note

    This approach is not recommended, as it produces unstable weight estimates. It is
    recommended to instead use [`BootstrappedPseudoBMA`](@ref) to stabilize the weights
    or [`Stacking`](@ref). For details, see [^YaoVehtari2018].

[^YaoVehtari2018]: Yuling Yao, Aki Vehtari, Daniel Simpson, and Andrew Gelman.
    Using Stacking to Average Bayesian Predictive Distributions.
    2018. Bayesian Analysis. 13, 3, 917–1007.
    doi: [10.1214/17-BA1091](https://doi.org/10.1214/17-BA1091)
    arXiv: [1704.02030](https://arxiv.org/abs/1704.02030)

See also: [`Stacking`](@ref)
"""
@kwdef struct PseudoBMA <: AbstractModelWeightsMethod
    regularize::Bool = false
end

function model_weights(method::PseudoBMA, elpd_results)
    elpds = map(elpd_results) do result
        est = elpd_estimates(result)
        method.regularize || return est.elpd
        return est.elpd - est.elpd_mcse / 2
    end
    return _akaike_weights(elpds)
end

"""
$(TYPEDEF)

Model weighting method using pseudo Bayesian Model Averaging using Akaike-type weighting
with the Bayesian bootstrap (pseudo-BMA+)[^YaoVehtari2018].

The Bayesian bootstrap stabilizes the model weights.

    BootstrappedPseudoBMA(; rng=Random.default_rng(), samples=1_000, alpha=1)
    BootstrappedPseudoBMA(rng, samples, alpha)

Construct the method.

$(TYPEDFIELDS)

See also: [`Stacking`](@ref)

[^YaoVehtari2018]: Yuling Yao, Aki Vehtari, Daniel Simpson, and Andrew Gelman.
    Using Stacking to Average Bayesian Predictive Distributions.
    2018. Bayesian Analysis. 13, 3, 917–1007.
    doi: [10.1214/17-BA1091](https://doi.org/10.1214/17-BA1091)
    arXiv: [1704.02030](https://arxiv.org/abs/1704.02030)
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
    return _assimilar(elpd_results, weights_mean)
end

function _model_weights_bootstrap!(w, elpd_mean, α, rng, α_dist, ic_mat)
    Random.rand!(rng, α_dist, α)
    mul!(elpd_mean, ic_mat', α)
    elpd_mean .*= length(α)
    akaike_weights!(w, elpd_mean)
    return w
end

"""
$(TYPEDEF)

Model weighting using stacking of predictive distributions[^YaoVehtari2018].

    Stacking(; optimizer=Optim.LBFGS(), options=Optim.Options()
    Stacking(optimizer[, options])

Construct the method, optionally customizing the optimization.

$(TYPEDFIELDS)

See also: [`BootstrappedPseudoBMA`](@ref)

[^YaoVehtari2018]: Yuling Yao, Aki Vehtari, Daniel Simpson, and Andrew Gelman.
    Using Stacking to Average Bayesian Predictive Distributions.
    2018. Bayesian Analysis. 13, 3, 917–1007.
    doi: [10.1214/17-BA1091](https://doi.org/10.1214/17-BA1091)
    arXiv: [1704.02030](https://arxiv.org/abs/1704.02030)
"""
Base.@kwdef struct Stacking{O<:Optim.AbstractOptimizer} <: AbstractModelWeightsMethod
    """The optimizer to use for the optimization of the weights. The optimizer must support
    projected gradient optimization viae a `manifold` field."""
    optimizer::O = DEFAULT_STACKING_OPTIMIZER
    """The Optim options to use for the optimization of the weights."""
    options::Optim.Options = Optim.Options()

    function Stacking(
        optimizer::Optim.AbstractOptimizer, options::Optim.Options=Optim.Options()
    )
        hasfield(typeof(optimizer), :manifold) ||
            throw(ArgumentError("The optimizer must have a `manifold` field."))
        _optimizer = Setfield.@set optimizer.manifold = Optim.Sphere()
        return new{typeof(_optimizer)}(_optimizer, options)
    end
end

function model_weights(method::Stacking, elpd_pairs)
    ic_mat = _elpd_matrix(elpd_pairs)
    exp_ic_mat = exp.(ic_mat)
    _, weights = _model_weights_stacking(exp_ic_mat, method.optimizer, method.options)
    return _assimilar(elpd_pairs, weights)
end
function _model_weights_stacking(exp_ic_mat, optimizer, options)
    # set up optimization objective
    objective = InplaceStackingOptimObjective(optimizer.manifold, exp_ic_mat)

    # set up initial point on optimization manifold
    w0 = similar(exp_ic_mat, axes(exp_ic_mat, 2))
    fill!(w0, 1//length(w0))
    x0 = _initial_point(objective, w0)

    # optimize
    sol = Optim.optimize(Optim.only_fg!(objective), x0, optimizer, options)

    # check convergence
    Optim.converged(sol) ||
        @warn "Optimization of stacking weights failed to converge after $(Optim.iterations(sol)) iterations."

    # return solution and weights
    w = _final_point(objective, sol.minimizer)
    return sol, w
end

function _elpd_matrix(elpd_results)
    elpd_values = map(elpd_results) do result
        return vec(elpd_estimates(result; pointwise=true).elpd)
    end
    elpd_mat = reduce(hcat, elpd_values)
    return elpd_mat
end

struct InplaceStackingOptimObjective{M<:Optim.Manifold,E,C}
    manifold::M
    exp_ic_mat::E
    cache::C
end
function InplaceStackingOptimObjective(manifold, exp_ic_mat)
    return InplaceStackingOptimObjective(manifold, exp_ic_mat, nothing)
end

# Optimize on the probability simplex by converting the problem to optimization on the unit
# sphere, optimizing with projected gradients, and mapping the solution back to the sphere.
# When the objective function on the simplex is convex, each global minimizer on the sphere
# maps to the global minimizer on the simplex, but the optimization manifold is simple, and
# no inequality constraints exist.
# Q Li, D McKenzie, W Yin. "From the simplex to the sphere: faster constrained optimization
# using the Hadamard parametrization." Inf. Inference. 12.3 (2023): iaad017.
# doi: 10.1093/imaiai/iaad017. arXiv: 2112.05273
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

# if ∑xᵢ² = 1, then if wᵢ = xᵢ², then w is on the probability simplex
_sphere_to_simplex(x) = x .^ 2
_simplex_to_sphere(x) = sqrt.(x)
function _∇sphere_to_simplex!(∂x, x)
    ∂x .*= 2 .* x
    return ∂x
end
