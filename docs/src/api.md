# [API Overview](@id api)

## [Plots](@id plots-api)

| Name                     | Description                                                                              |
|:------------------------ |:---------------------------------------------------------------------------------------- |
| [`plot_autocorr`](@ref)  | Bar plot of the autocorrelation function for a sequence of data.                         |
| [`plot_compare`](@ref)   | Summary plot for model comparison.                                                       |
| [`plot_density`](@ref)   | Generate KDE plots for continuous variables and histograms for discrete ones.            |
| [`plot_dist`](@ref)      | Plot distribution as histogram or kernel density estimates.                              |
| [`plot_elpd`](@ref)      | Plot a scatter or hexbin matrix of the sampled parameters.                               |
| [`plot_energy`](@ref)    | Plot energy transition distribution and marginal energy distribution in HMC algorithms.  |
| [`plot_ess`](@ref)       | Plot quantile, local or evolution of effective sample sizes (ESS).                       |
| [`plot_forest`](@ref)    | Forest plot to compare credible intervals from a number of distributions.                |
| [`plot_hpd`](@ref)       | Plot hpd intervals for regression data.                                                  |
| [`plot_joint`](@ref)     | Plot a scatter or hexbin of two variables with their respective marginals distributions. |
| [`plot_kde`](@ref)       | 1D or 2D KDE plot taking into account boundary conditions.                               |
| [`plot_khat`](@ref)      | Plot Pareto tail indices.                                                                |
| [`plot_loo_pit`](@ref)   | Plot Leave-One-Out (LOO) probability integral transformation (PIT) predictive checks.    |
| [`plot_mcse`](@ref)      | Plot quantile, local or evolution of effective sample sizes (ESS).                       |
| [`plot_pair`](@ref)      | Plot a scatter or hexbin matrix of the sampled parameters.                               |
| [`plot_parallel`](@ref)  | Plot parallel coordinates plot showing posterior points with and without divergences.    |
| [`plot_posterior`](@ref) | Plot Posterior densities in the style of John K.                                         |
| [`plot_ppc`](@ref)       | Plot for posterior predictive checks.                                                    |
| [`plot_rank`](@ref)      | Plot rank order statistics of chains.                                                    |
| [`plot_trace`](@ref)     | Plot distribution (histogram or kernel density estimates) and sampled values.            |
| [`plot_violin`](@ref)    | Plot posterior of traces as violin plot.                                                 |

## [Stats](@id stats-api)

| Name                     | Description                                                                     |
|:------------------------ |:------------------------------------------------------------------------------- |
| [`summarystats`](@ref)   | Compute summary statistics on an `InferenceData`                                |
| [`hpd`](@ref)            | Calculate highest posterior density (HPD) of array for given credible_interval. |
| [`loo`](@ref)            | Pareto-smoothed importance sampling leave-one-out (LOO) cross-validation.       |
| [`loo_pit`](@ref)        | Compute leave-one-out probability integral transform (PIT) values.              |
| [`psislw`](@ref)         | Pareto smoothed importance sampling (PSIS).                                     |
| [`r2_score`](@ref)       | $R^2$ for Bayesian regression models.                                           |
| [`waic`](@ref)           | Calculate the widely available information criterion (WAIC).                    |
| [`compare`](@ref)        | Compare models based on WAIC or LOO cross-validation.                           |

## [Diagnostics](@id diagnostics-api)

| Name             | Description                                                              |
|:---------------- |:------------------------------------------------------------------------ |
| [`bfmi`](@ref)   | Calculate the estimated Bayesian fraction of missing information (BFMI). |
| [`geweke`](@ref) | Compute $z$-scores for convergence diagnostics.                          |
| [`ess`](@ref)    | Calculate estimate of the effective sample size (ESS).                   |
| [`rhat`](@ref)   | Compute estimate of rank normalized split-$\hat{R}$ for a set of traces. |
| [`mcse`](@ref)   | Calculate Markov Chain Standard Error statistic (MCSE).                  |

## [Stats utils](@id statsutils-api)

| Name                        | Description                                                          |
|:--------------------------- |:-------------------------------------------------------------------- |
| [`autocov`](@ref)           | Compute autocovariance estimates for every lag for the input array.  |
| [`autocorr`](@ref)          | Compute autocorrelation using FFT for every lag for the input array. |
| [`make_ufunc`](@ref)        | Make ufunc from a function taking 1D array input.                    |
| [`wrap_xarray_ufunc`](@ref) | Wrap `make_ufunc` with `xarray.apply_ufunc`.                         |

## [Data](@id data-api)

| Name                                | Description                                        |
|:----------------------------------- |:-------------------------------------------------- |
| [`InferenceData`](@ref)             | Container for inference data storage using xarray. |
| [`convert_to_inference_data`](@ref) | Convert a supported object to an `InferenceData`.  |
| [`load_arviz_data`](@ref)           | Load a local or remote pre-made dataset.           |
| [`to_netcdf`](@ref)                 | Save dataset as a netcdf file.                     |
| [`from_netcdf`](@ref)               | Load netcdf file back into an `InferenceData`.     |
| [`from_dict`](@ref)                 | Convert `Dict` data into an `InferenceData`.       |
| [`from_cmdstan`](@ref)              | Convert `CmdStan` data into an `InferenceData`.    |
| [`from_mcmcchains`](@ref)           | Convert `MCMCChains` data into an `InferenceData`. |
| [`concat`](@ref)                    | Concatenate `InferenceData` objects.               |
| [`concat!`](@ref)                   | Concatenate `InferenceData` objects in-place.      |

## [Utils](@id utils-api)

| Name                               | Description                          |
|:---------------------------------- |:------------------------------------ |
| [`with_interactive_backend`](@ref) | Change plotting backend temporarily. |

## [rcParams](@id rcparams-api)

| Name                      | Description                                              |
|:------------------------- |:-------------------------------------------------------- |
| [`with_rc_context`](@ref) | Change ArviZ's matplotlib-style rc settings temporarily. |
