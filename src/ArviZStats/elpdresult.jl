"""
$(TYPEDEF)

An abstract type representing the result of an ELPD computation.

Every subtype stores estimates of both the expected log predictive density (`elpd`) and the
effective number of parameters `p`, as well as standard errors and pointwise estimates of
each, from which other relevant estimates can be computed.

Subtypes implement the following functions:
- [`elpd_estimates`](@ref)
- [`information_criterion`](@ref)
"""
abstract type AbstractELPDResult end

function _print_elpd_estimates(
    io::IO, ::MIME"text/plain", r::AbstractELPDResult; sigdigits_se=2
)
    estimates = elpd_estimates(r)
    elpd, elpd_mcse = estimates.elpd, estimates.elpd_mcse
    p, p_mcse = estimates.p, estimates.p_mcse
    table = (; Estimate=[elpd, p], SE=[elpd_mcse, p_mcse])
    formatters = function (v, i, j)
        sigdigits = j == 1 ? sigdigits_matching_error(v, table.SE[i]) : sigdigits_se
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

function _lpd_pointwise(log_likelihood, dims)
    ndraws = prod(Base.Fix1(size, log_likelihood), dims)
    lpd = LogExpFunctions.logsumexp(log_likelihood; dims)
    T = eltype(lpd)
    return dropdims(lpd; dims) .- log(T(ndraws))
end

function _elpd_estimates_from_pointwise(pointwise)
    elpd, elpd_mcse = _sum_and_se(pointwise.elpd)
    p, p_mcse = _sum_and_se(pointwise.p)
    return (; elpd, elpd_mcse, p, p_mcse)
end
