using RCall

r_loo_installed() = !isempty(rcopy(R"system.file(package='loo')"))

# R loo with our API
function loo_r(log_likelihood; reff=nothing)
    R"require('loo')"
    if reff === nothing
        reff = rcopy(R"loo::relative_eff(exp($(log_likelihood)))")
    end
    result = R"loo::loo($log_likelihood, r_eff=$reff)"
    estimates = rcopy(R"$(result)$estimates")
    estimates = (
        elpd=estimates[1, 1],
        elpd_mcse=estimates[1, 2],
        p=estimates[2, 1],
        p_mcse=estimates[2, 2],
    )
    pointwise = rcopy(R"$(result)$pointwise")
    pointwise = (
        elpd=pointwise[:, 1],
        elpd_mcse=pointwise[:, 2],
        p=pointwise[:, 3],
        reff=reff,
        pareto_shape=pointwise[:, 5],
    )
    return (; estimates, pointwise)
end

# R loo with our API
function waic_r(log_likelihood)
    R"require('loo')"
    result = R"loo::waic($log_likelihood)"
    estimates = rcopy(R"$(result)$estimates")
    estimates = (
        elpd=estimates[1, 1],
        elpd_mcse=estimates[1, 2],
        p=estimates[2, 1],
        p_mcse=estimates[2, 2],
    )
    pointwise = rcopy(R"$(result)$pointwise")
    pointwise = (elpd=pointwise[:, 1], p=pointwise[:, 2])
    return (; estimates, pointwise)
end

function _isequal(x::ModelComparisonResult, y::ModelComparisonResult)
    return Tables.columntable(x) == Tables.columntable(y)
end

_isapprox(x::AbstractArray, y::AbstractArray; kwargs...) = isapprox(x, y; kwargs...)
_isapprox(x, y; kwargs...) = all(isapprox.(x, y; kwargs...))
