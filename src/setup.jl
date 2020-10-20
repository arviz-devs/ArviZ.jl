import_arviz() = _import_dependency("arviz", "arviz"; channel = "conda-forge")

arviz_version() = VersionNumber(arviz.__version__)

function check_needs_update(; update = true)
    if arviz_version() < _min_arviz_version
        @warn "ArviZ.jl only officially supports arviz version $(_min_arviz_version) or " *
              "greater but found version $(arviz_version())."
        if update
            if update_arviz()
                # yay, but we still already imported the old version
                msg = """
                Please rebuild ArviZ.jl with `using Pkg; Pkg.build("ArviZ")` and re-launch Julia
                to continue.
                """
            else
                msg = """
                Could not automatically update arviz. Please manually update arviz, rebuild
                ArviZ.jl with `using Pkg; Pkg.build("ArviZ")`, and then re-launch Julia to
                continue.
                """
            end
            @warn msg
        end
    end
    return nothing
end

function check_needs_rebuild()
    if arviz_version() != _precompile_arviz_version
        msg = """
        ArviZ.jl was built using arviz version $(_precompile_arviz_version) but loaded with
        version $(arviz_version()). Please recompile with `using Pkg; Pkg.build("ArviZ")`
        and re-launch Julia to continue.
        """
        @warn msg
    end
    return nothing
end

function initialize_arviz()
    ispynull(arviz) || return
    copy!(arviz, import_arviz())
    check_needs_update(update = true)
    check_needs_rebuild()

    pytype_mapping(arviz.InferenceData, InferenceData)

    # pytypemap-ing RcParams produces a Dict
    copy!(_rcParams, py"$(arviz).rcparams.rcParams"o)

    # use 1-based indexing by default within arviz
    rcParams["data.index_origin"] = 1

    # handle Bokeh showing ourselves
    rcParams["plot.bokeh.show"] = false

    initialize_xarray()
    initialize_numpy()
    return nothing
end

function initialize_xarray()
    ispynull(xarray) || return
    copy!(xarray, _import_dependency("xarray", "xarray"; channel = "conda-forge"))
    _import_dependency("dask", "dask"; channel = "conda-forge")
    pytype_mapping(xarray.Dataset, Dataset)
    return nothing
end

function initialize_numpy()
    # Trigger NumPy initialization, see https://github.com/JuliaPy/PyCall.jl/issues/744
    PyObject([true])
    return nothing
end

function initialize_pandas()
    ispynull(pandas) || return
    copy!(pandas, _import_dependency("pandas", "pandas"; channel = "conda-forge"))
    return nothing
end

function update_arviz()
    # updating arviz can change other packages, so we always ask for permission
    if _using_conda() && _isyes(Base.prompt("Try updating arviz using conda? [Y/n]"))
        # this syntax isn't officially supported, but it works (for now)
        try
            Conda.add("arviz>=$_min_arviz_version"; channel = "conda-forge")
            return true
        catch e
            println(e.msg)
        end
    end
    if _has_pip() && _isyes(Base.prompt("Try updating arviz using pip? [Y/n]"))
        # can't specify version lower bound, so update to latest
        try
            run(`$(PyCall.pyprogramname) -m pip install --upgrade -- arviz`)
            return true
        catch e
            println(e.msg)
        end
    end
    return false
end

function _import_dependency(modulename, pkgname; channel = nothing)
    _has_pymodule(modulename) && return pyimport(modulename)
    return if _using_conda()
        # auto-installing with conda is safe and convenient
        if channel === nothing
            pyimport_conda(modulename, pkgname)
        else
            pyimport_conda(modulename, pkgname, channel)
        end
    elseif _has_pip() && _isyes(Base.prompt("Try installing $pkgname using pip? [Y/n]"))
        # installing with pip is riskier, so we ask for permission
        run(`$(PyCall.pyprogramname) -m pip install -- $pkgname`)
        pyimport(modulename)
    else
        error("Dependency $modulename cannot be imported. Install manually to continue.")
    end
end

_isyes(s) = isempty(s) || lowercase(strip(s)) ∈ ("y", "yes")

_using_conda() = PyCall.conda

_has_pip() = ispynull(pyimport_e("pip"))

_has_pymodule(modulename) = !ispynull(pyimport_e(modulename))
