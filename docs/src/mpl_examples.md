# Matplotlib Example Gallery

!!! note
    These examples are adapted from [ArviZ's matplotlib gallery](https://arviz-devs.github.io/arviz/examples/index.html#matplotlib).

## Autocorrelation Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

data = load_arviz_data("centered_eight")
plot_autocorr(data; var_names = ("tau", "mu"))

gcf()
```

See [`plot_autocorr`](@ref)

## Compare Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

model_compare = compare(Dict(
    "Centered 8 schools" => load_arviz_data("centered_eight"),
    "Non-centered 8 schools" => load_arviz_data("non_centered_eight"),
))
plot_compare(model_compare; figsize = (12, 4))

gcf()
```

See [`compare`](@ref), [`plot_compare`](@ref)

## Density Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

centered_data = load_arviz_data("centered_eight")
non_centered_data = load_arviz_data("non_centered_eight")
plot_density(
    [centered_data, non_centered_data];
    data_labels = ["Centered", "Non Centered"],
    var_names = ["theta"],
    shade = 0.1,
)

gcf()
```

See [`plot_density`](@ref)

## Dist Plot

```@example
using Random
using Distributions
using PyPlot
figure() #hide
using ArviZ

Random.seed!(308)

ArviZ.use_style("arviz-darkgrid")

a = rand(Poisson(4), 1000)
b = rand(Normal(0, 1), 1000)
_, ax = plt.subplots(1, 2; figsize = (10, 4))
plot_dist(a; color = "C1", label = "Poisson", ax = ax[1])
plot_dist(b; color = "C2", label = "Gaussian", ax = ax[2])

gcf()
```

See [`plot_dist`](@ref)

## ELPD Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

d1 = load_arviz_data("centered_eight")
d2 = load_arviz_data("non_centered_eight")
plot_elpd(Dict("Centered eight" => d1, "Non centered eight" => d2); xlabels = true)

gcf()
```

See [`plot_elpd`](@ref)


## Energy Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

data = load_arviz_data("centered_eight")
plot_energy(data; figsize = (12, 8))

gcf()
```

See [`plot_energy`](@ref)


## ESS Quantile Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

idata = load_arviz_data("radon")
plot_ess(idata; var_names = ["b"], kind = "evolution")

gcf()
```

See [`plot_ess`](@ref)

## ESS Local Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

idata = load_arviz_data("non_centered_eight")
plot_ess(
    idata;
    var_names = ["mu"],
    kind = "local",
    marker = "_",
    ms = 20,
    mew = 2,
    rug = true,
)

gcf()
```

See [`plot_ess`](@ref)

## ESS Quantile Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

idata = load_arviz_data("radon")
plot_ess(idata; var_names = ["sigma_y"], kind = "quantile", color = "C4")

gcf()
```

See [`plot_ess`](@ref)

## Forest Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

centered_data = load_arviz_data("centered_eight")
non_centered_data = load_arviz_data("non_centered_eight")
plot_forest(
    [centered_data, non_centered_data];
    model_names = ["Centered", "Non Centered"],
    var_names = ["mu"],
)
title("Estimated theta for eight schools model")

gcf()
```

See [`plot_forest`](@ref)

## Ridge Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

rugby_data = load_arviz_data("rugby")
plot_forest(
    rugby_data;
    kind = "ridgeplot",
    var_names = ["defs"],
    linewidth = 4,
    combined = true,
    ridgeplot_overlap = 1.5,
    colors = "blue",
    figsize = (9, 4),
)
title("Relative defensive strength\nof Six Nation rugby teams")

gcf()
```

See [`plot_forest`](@ref)

## Plot HDI

```@example
using Random
using PyPlot
figure() #hide
using ArviZ

Random.seed!(308)

ArviZ.use_style("arviz-darkgrid")

x_data = randn(100)
y_data = 2 .+ x_data .* 0.5
y_data_rep = 0.5 .* randn(200, 100) .+ transpose(y_data)
plot(x_data, y_data; color = "C6")
plot_hdi(x_data, y_data_rep; color = "k", plot_kwargs = Dict("ls" => "--"))

gcf()
```

See [`plot_hdi`](@ref)

## Joint Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

data = load_arviz_data("non_centered_eight")
plot_pair(
    data;
    var_names = ["theta"],
    coords = Dict("school" => ["Choate", "Phillips Andover"]),
    kind = "hexbin",
    marginals = true,
    figsize = (10, 10),
)

gcf()
```

See [`plot_pair`](@ref)

## KDE Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

data = load_arviz_data("centered_eight")

## Combine different posterior draws from different chains
obs = data.posterior_predictive["obs"].values
size_obs = size(obs)
y_hat = reshape(obs, prod(size_obs[1:2]), size_obs[3:end]...)

plot_kde(
    y_hat;
    label = "Estimated Effect\n of SAT Prep",
    rug = true,
    plot_kwargs = Dict("linewidth" => 2, "color" => "black"),
    rug_kwargs = Dict("color" => "black"),
)

gcf()
```

See [`plot_kde`](@ref)

## 2d KDE

```@example
using Random
using PyPlot
figure() #hide
using ArviZ

Random.seed!(308)

ArviZ.use_style("arviz-darkgrid")

plot_kde(rand(100), rand(100))

gcf()
```

See [`plot_kde`](@ref)

## KDE Quantiles Plot

```@example
using Random
using Distributions
using PyPlot
figure() #hide
using ArviZ

Random.seed!(308)

ArviZ.use_style("arviz-darkgrid")

dist = rand(Beta(rand(Uniform(0.5, 10)), 5), 1000)
plot_kde(dist; quantiles = [0.25, 0.5, 0.75])

gcf()
```

See [`plot_kde`](@ref)

## Pareto Shape Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

idata = load_arviz_data("radon")
loo_data = loo(idata; pointwise = true)
plot_khat(loo_data; show_bins = true)

gcf()
```

See [`loo`](@ref), [`plot_khat`](@ref)

## LOO-PIT ECDF Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

idata = load_arviz_data("radon")
log_like = transpose(idata.sample_stats.log_likelihood.sel(chain = 0).values)
log_weights = psislw(-log_like)[1]
plot_loo_pit(idata; y = "y_like", log_weights = log_weights, ecdf = true, color = "maroon")

gcf()
```

See [`psislw`](@ref), [`plot_loo_pit`](@ref)

## LOO-PIT Overlay Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

idata = load_arviz_data("non_centered_eight")
plot_loo_pit(idata = idata, y = "obs", color = "indigo")

gcf()
```

See [`plot_loo_pit`](@ref)

## Quantile Monte Carlo Standard Error Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

data = load_arviz_data("centered_eight")
plot_mcse(data; var_names = ["tau", "mu"], rug = true, extra_methods = true)

gcf()
```

See [`plot_mcse`](@ref)

## Quantile MCSE Errobar Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

data = load_arviz_data("radon")
plot_mcse(data; var_names = ["sigma_a"], color = "C4", errorbar = true)

gcf()
```

See [`plot_mcse`](@ref)

## Pair Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

centered = load_arviz_data("centered_eight")
coords = Dict("school" => ["Choate", "Deerfield"])
plot_pair(
    centered;
    var_names = ["theta", "mu", "tau"],
    coords = coords,
    divergences = true,
    textsize = 22,
)

gcf()
```

See [`plot_pair`](@ref)

## Hexbin Pair Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

centered = load_arviz_data("centered_eight")
coords = Dict("school" => ["Choate", "Deerfield"])
plot_pair(
    centered;
    var_names = ["theta", "mu", "tau"],
    kind = "hexbin",
    coords = coords,
    colorbar = true,
    divergences = true,
)

gcf()
```

See [`plot_pair`](@ref)

## KDE Pair Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

centered = load_arviz_data("centered_eight")
coords = Dict("school" => ["Choate", "Deerfield"])
plot_pair(
    centered;
    var_names = ["theta", "mu", "tau"],
    kind = "kde",
    coords = coords,
    divergences = true,
    textsize = 22,
)

gcf()
```

See [`plot_pair`](@ref)

## Point Estimate Pair Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

centered = load_arviz_data("centered_eight")
coords = Dict("school" => ["Choate", "Deerfield"])
plot_pair(
    centered;
    var_names = ["mu", "theta"],
    kind = ["scatter", "kde"],
    kde_kwargs = Dict("fill_last" => false),
    marginals = true,
    coords = coords,
    point_estimate = "median",
    figsize = (10, 8),
)

gcf()
```

See [`plot_pair`](@ref)

## Parallel Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

data = load_arviz_data("centered_eight")
ax = plot_parallel(data; var_names = ["theta", "tau", "mu"])
ax.set_xticklabels(ax.get_xticklabels(); rotation = 70)
draw()

gcf()
```

See [`plot_parallel`](@ref)

## Posterior Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

data = load_arviz_data("centered_eight")
coords = Dict("school" => ["Choate"])
plot_posterior(data; var_names = ["mu", "theta"], coords = coords, rope = (-1, 1))

gcf()
```

See [`plot_posterior`](@ref)

## Posterior Predictive Check Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

data = load_arviz_data("non_centered_eight")
plot_ppc(
    data;
    data_pairs = Dict("obs" => "obs"),
    alpha = 0.03,
    figsize = (12, 6),
    textsize = 14,
)

gcf()
```

See [`plot_ppc`](@ref)

## Posterior Predictive Check Cumulative Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

data = load_arviz_data("non_centered_eight")
plot_ppc(data; alpha = 0.3, kind = "cumulative", figsize = (12, 6), textsize = 14)

gcf()
```

See [`plot_ppc`](@ref)

## Rank Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

data = load_arviz_data("centered_eight")
plot_rank(data; var_names = ("tau", "mu"))

gcf()
```

See [`plot_rank`](@ref)

## Trace Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

data = load_arviz_data("non_centered_eight")
plot_trace(data; var_names = ("tau", "mu"))

gcf()
```

See [`plot_trace`](@ref)

## Violin Plot

```@example
using PyPlot
figure() #hide
using ArviZ

ArviZ.use_style("arviz-darkgrid")

data = load_arviz_data("non_centered_eight")
plot_violin(data; var_names = ["mu", "tau"])

gcf()
```

See [`plot_violin`](@ref)

## Styles

```@example
using PyPlot
figure() #hide
using PyCall
using Distributions
using ArviZ

x = range(0, 1; length = 100)
dist = pdf.(Beta(2, 5), x)

style_list = [
    "default",
    ["default", "arviz-colors"],
    "arviz-darkgrid",
    "arviz-whitegrid",
    "arviz-white",
]

fig = figure(figsize = (12, 12))
for (idx, style) in enumerate(style_list)
    @pywith plt.style.context(style) begin
        ax = fig.add_subplot(3, 2, idx; label = idx)
        for i = 0:9
            ax.plot(x, dist .- i, "C$i", label = "C$i")
        end
        ax.set_title(style)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)"; rotation = 0, labelpad = 15)
        ax.legend(bbox_to_anchor = (1, 1))
        draw()
    end
end
tight_layout()

gcf()
```
