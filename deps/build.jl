using Conda

# try to install scipy with pip if not yet installed
# temporary workaround for https://github.com/arviz-devs/ArviZ.jl/issues/188
Conda.pip_interop(true)
Conda.pip("uninstall -y --no-deps", "scipy")
Conda.pip("install", "scipy")
