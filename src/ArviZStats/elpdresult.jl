"""
$(TYPEDEF)

An abstract type representing the result of an ELPD computation.

Every subtype stores estimates of both the expected log predictive density (ELPD) and the
log pointwise predictive density (LPD), as well as pointwise estimates of each, from which
other relevant estimates can be computed.

Subtypes implement the following functions:
- [`elpd_estimates`](@ref)
- [`effective_number_of_parameters`](@ref)
- [`information_criterion`](@ref)
"""
abstract type AbstractELPDResult end

function _print_elpd_estimates(
    io::IO, ::MIME"text/plain", r::AbstractELPDResult; sigdigits_se=2
)
    estimates = elpd_estimates(r)
    elpd, elpd_mcse = estimates.elpd, estimates.elpd_mcse
    p_pointwise = effective_number_of_parameters(r; pointwise=true)
    p, p_mcse = _sum_and_se(p_pointwise)
    table = (; Estimate=[elpd, p], SE=[elpd_mcse, p_mcse])
    formatters = function (v, i, j)
        sigdigits = j == 1 ? _sigdigits_matching_error(v, table.SE[i]) : sigdigits_se
        return sprint(Printf.format, Printf.Format("%.$(sigdigits)g"), v)
    end
    PrettyTables.pretty_table(
        io,
        table;
        show_subheader=false,
        row_labels=["elpd", "p"],
        hlines=:none,
        vlines=:none,
        formatters,
    )
    return nothing
end

"""
    $(FUNCTIONNAME)(result::AbstractELPDResult; pointwise=false) -> (; elpd, elpd_mcse, lpd)

Return the (E)LPD estimates from the `result`.
"""
function elpd_estimates end

"""
    $(FUNCTIONNAME)(elpd, scale::Symbol)

Compute the information criterion for the given `scale` from the `elpd` estimate.

`scale` must be one of $(keys(INFORMATION_CRITERION_SCALES)).

See also: [`effective_number_of_parameters`](@ref), [`loo`](@ref), [`waic`](@ref)
"""
function information_criterion(estimates, scale::Symbol)
    scale_value = INFORMATION_CRITERION_SCALES[scale]
    return scale_value * estimates.elpd
end

"""
    $(FUNCTIONNAME)(result::AbstractELPDResult, scale::Symbol; pointwise=false)

Compute information criterion for the given `scale` from the existing ELPD `result`.

`scale` must be one of $(keys(INFORMATION_CRITERION_SCALES)).

If `pointwise=true`, then pointwise estimates are returned.
"""
function information_criterion(
    result::AbstractELPDResult, scale::Symbol; pointwise::Bool=false
)
    return information_criterion(elpd_estimates(result; pointwise), scale)
end

"""
    $(FUNCTIONNAME)(elpd, lpd, scale::Symbol)

Compute the effective number of parameters ``p`` for the given `scale` from `elpd` and `lpd`
estimates.

`scale` must be one of $(keys(INFORMATION_CRITERION_SCALES)).

See also: [`information_criterion`](@ref), [`loo`](@ref), [`waic`](@ref)
"""
effective_number_of_parameters(elpd, lpd) = lpd - elpd

"""
    $(FUNCTIONNAME)(result::AbstractELPDResult; pointwise=false) -> Real

Compute the effective number of parameters ``p`` from the ELPD `result`.

If `pointwise=true`, then pointwise estimates are returned.
"""
function effective_number_of_parameters(result::AbstractELPDResult; pointwise::Bool=false)
    estimates = elpd_estimates(result; pointwise)
    return effective_number_of_parameters(estimates.elpd, estimates.lpd)
end

function _lpd_pointwise(log_likelihood, dims)
    ndraws = prod(Base.Fix1(size, log_likelihood), dims)
    lpd = LogExpFunctions.logsumexp(log_likelihood; dims)
    T = eltype(lpd)
    return dropdims(lpd; dims) .- log(T(ndraws))
end

function _elpd_estimates_from_pointwise(pointwise)
    elpd, elpd_mcse = _sum_and_se(pointwise.elpd)
    lpd = sum(pointwise.lpd)
    return (; elpd, elpd_mcse, lpd)
end
