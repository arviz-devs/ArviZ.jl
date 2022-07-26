using PyCall

# try to install scipy with pip if not yet installed
# temporary workaround for https://github.com/arviz-devs/ArviZ.jl/issues/188
run(PyCall.python_cmd(`-m pip install --force-reinstall -- scipy`))
