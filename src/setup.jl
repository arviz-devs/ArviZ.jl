import_arviz() = pyimport_conda("arviz", "arviz", "conda-forge")

arviz_version() = VersionNumber(arviz.__version__)

function check_version()
    if arviz_version() < _min_arviz_version
        @warn "ArviZ.jl only officially supports arviz version $(_min_arviz_version) or greater but found version $(arviz_version()). Please update."
    end
    return nothing
end

function initialize_arviz()
    ispynull(arviz) || return
    copy!(arviz, import_arviz())
    if arviz_version() != _precompile_arviz_version
        @warn "ArviZ.jl was precompiled using arviz version $(_precompile_version) but loaded with version $(arviz_version()). Please recompile with `using Pkg; Pkg.build('ArviZ')`."
    end

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
    copy!(xarray, pyimport_conda("xarray", "xarray", "conda-forge"))
    pyimport_conda("dask", "dask", "conda-forge")
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
    copy!(pandas, pyimport_conda("pandas", "pandas", "conda-forge"))
    return nothing
end
