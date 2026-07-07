### A Pluto.jl notebook ###
# v0.19.10

using Markdown
using InteractiveUtils

# ╔═╡ e5e15f39-027c-42aa-a765-588ec1cd0b63
using Pkg, InteractiveUtils

# ╔═╡ b1784f52-0cf2-11ed-32ba-69a8f63b48a9
# Remove this cell to use release versions of dependencies 
# hideall
let
    docs_dir = dirname(@__DIR__)
    pkg_dir = dirname(docs_dir)

    Pkg.activate(docs_dir)
    Pkg.develop(; path=pkg_dir)
    Pkg.instantiate()
end;

# ╔═╡ 34ee7907-49e7-4e32-9bd5-1c5d490a13d8
begin
    using ArviZ, DimensionalData, DataFrames, Statistics, AlgebraOfGraphics, CairoMakie
    using AlgebraOfGraphics: density
    set_aog_theme!()
end;

# ╔═╡ 41b1ddca-20dc-4821-9cec-d7192ef34795
using PlutoUI

# ╔═╡ 70f66296-349c-48a5-9b83-cc5c6cdbd514
md"""
# Creating custom plots
"""

# ╔═╡ 47c7a1fd-0862-49d5-879d-d59d321ce014
md"""
While ArviZ includes many plotting functions for visualizing the data stored in `InferenceData` objects, you will often need to construct custom plots, or you may want to tweak some of our plots in your favorite plotting package.

In this tutorial, we will show you a few useful techniques you can use to construct these plots using Julia's plotting packages.
For demonstration purposes, we'll use Makie.jl and AlgebraOfGraphics.jl, which can consume `Dataset` objects since they implement the Tables interface.
However, we could just as easily have used StatsPlots.jl.
"""

# ╔═╡ 7bdff7ba-408c-4893-9857-7cf328301a34
md"""
We'll start by loading some draws from an implementation of the non-centered parameterization of the 8 schools model.
In this parameterization, the model has some sampling issues.
"""

# ╔═╡ cbd81558-e355-4e7a-ba51-0ba9299cb558
idata = load_arviz_data("centered_eight")

# ╔═╡ 898d83d1-cd3a-47a4-84b1-f2cf0f7bf959
idata.posterior

# ╔═╡ da4e989e-a08c-45af-8e55-20ac3ed58745
md"""
The plotting functions we'll be using interact with a tabular view of a `Dataset`.
Let's see what that view looks like for a `Dataset`:
"""

# ╔═╡ 52ef5ef5-2ea8-4e1a-8cd5-ba3aa5da6d9b
df = DataFrame(idata.posterior)

# ╔═╡ e77c2bc3-8c23-446d-9ec4-3a155aea23e9
md"""
The tabular view includes dimensions and variables as columns.

When variables with different dimensions are flattened into a tabular form, there's always some duplication of values.
As a simple case, note that `chain`, `draw`, and `school` all have repeated values in the above table.

In this case, `theta` has the `school` dimension, but `tau` doesn't, so the values of `tau` will be repeated in the table for each value of `school`.
"""

# ╔═╡ 0d625d42-726d-4cb3-8456-c54e1001df6d
df[df.school .== Ref("Choate"), :].tau == df[df.school .== Ref("Deerfield"), :].tau

# ╔═╡ 9b1a7256-c875-4266-b2fb-47c29f2b13c9
md"""
In our first example, this will be important.

Here, let's construct a trace plot.
Besides `idata`, all functions and types in the following cell are defined in AlgebraOfGraphics or Makie:
- `data(...)` indicates that the wrapped object implements the Tables interface
- `mapping` indicates how the data should be used. The symbols are all column names in the table, which for us are our variable names and dimensions.
- `visual` specifies how the data should be converted to a plot.
- `Lines` is a plot type defined in Makie.
- `draw` takes this combination and plots it.
"""

# ╔═╡ 0f4483b1-5820-45e2-8812-14550bff69e2
draw(
    data(idata.posterior.mu) *
    mapping(:draw, :mu; color=:chain => nonnumeric) *
    visual(Lines; alpha=0.8),
)

# ╔═╡ ee4ab468-dbf6-46af-a7c5-44b82c031c2c
md"""
Note the line `idata.posterior.mu`.
If we had just used `idata.posterior`, the plot would have looked more-or-less the same, but there would be artifacts due to `mu` being copied many times.
By selecting `mu` directly, all other dimensions are discarded, so each value of `mu` appears in the plot exactly once.

When examining an MCMC trace plot, we want to see a "fuzzy caterpillar".
Instead we see a few places where the Markov chains froze.
We can do the same for `theta` as well, but it's more useful here to separate these draws by `school`.
"""

# ╔═╡ 9b590c11-b15a-4199-8d9f-da8e1735fed2
draw(
    data(idata.posterior) *
    mapping(:draw, :theta; layout=:school, color=:chain => nonnumeric) *
    visual(Lines; alpha=0.8),
)

# ╔═╡ ca18a574-a5c6-4d83-a5d6-b5687f569522
md"""
Suppose we want to compare `tau` with `theta` for two different schools.
To do so, we use `InferenceData`s indexing syntax to subset the data.
"""

# ╔═╡ 746785a2-c472-467f-973a-d2390ec3e0bb
draw(
    data(idata[:posterior, school=At(["Choate", "Deerfield"])]) *
    mapping(:theta, :tau; color=:school) *
    density() *
    visual(Contour; levels=10),
)

# ╔═╡ e221c4b1-2256-4f97-bfd8-aeaf02fedc1b
md"""
We can also compare the density plots constructed from each chain for different schools.
"""

# ╔═╡ 663e5edb-a751-40e7-96ec-495685453515
draw(
    data(idata.posterior) *
    mapping(:theta; layout=:school, color=:chain => nonnumeric) *
    density(),
)

# ╔═╡ 6e10a67d-045a-4fa9-a7ad-90fa79015ea8
md"""
If we want to compare many schools in a single plot, an ECDF plot is more convenient.
"""

# ╔═╡ 9d8a23e1-4961-44c6-b272-afb8883df1d7
draw(
    data(idata.posterior) * mapping(:theta; color=:school => nonnumeric) * visual(ECDFPlot);
    axis=(; ylabel="probability"),
)

# ╔═╡ 7068ad2c-2260-4afc-b85d-229b2be6c207
md"""
So far we've just plotted data from one group, but we often want to combine data from multiple groups in one plot.
The simplest way to do this is to create the plot out of multiple layers.
Here we use this approach to plot the observations over the posterior predictive distribution.
"""

# ╔═╡ c860bb8d-c15c-4ac9-b2b7-143f6f12bf0b
draw(
    (data(idata.posterior_predictive) * mapping(:obs; layout=:school) * density()) +
    (data(idata.observed_data) * mapping(:obs, :obs => zero => ""; layout=:school)),
)

# ╔═╡ 6017222f-769b-4db7-9918-af70c792a158
md"""
Another option is to combine the groups into a single dataset.

Here we compare the prior and posterior.
Since the prior has 1 chain and the posterior has 4 chains, if we were to combine them into a table, the structure would need to be ragged.
This is not currently supported.

We can then either plot the two distributions separately as we did before, or we can compare a single chain from each group.
This is what we'll do here.
To concatenate the two groups, we introduce a new named dimension using `DimensionalData.Dim`.
"""

# ╔═╡ b072dd93-3076-4630-bb2a-b9d490045a78
draw(
    data(
        cat(
            idata.posterior[chain=[1]], idata.prior; dims=Dim{:group}([:posterior, :prior])
        )[:mu],
    ) *
    mapping(:mu; color=:group) *
    histogram(; bins=20) *
    visual(; alpha=0.8);
    axis=(; ylabel="probability"),
)

# ╔═╡ 82229ba7-101d-4fbc-96a9-ca56d899ebd5
md"""
From the trace plots, we suspected the geometry of this posterior was bad.
Let's highlight divergent transitions.
To do so, we merge `posterior` and `samplestats`, which can do with `merge` since they share no common variable names.
"""

# ╔═╡ 2a47f53a-a054-426d-b536-ccffcf62dd15
draw(
    data(merge(idata.posterior, idata.sample_stats)) * mapping(
        :theta,
        :tau;
        layout=:school,
        color=:diverging,
        markersize=:diverging => (d -> d ? 5 : 2),
    ),
)

# ╔═╡ 3c939c5a-90c6-4367-9a9b-2525796425ce
md"""
When we try building more complex plots, we may need to build new `Dataset`s from our existing ones.

One example of this is the corner plot.
To build this plot, we need to make a copy of `theta` with a copy of the `school` dimension.
"""

# ╔═╡ b3c8eed9-4083-457e-89dd-71d678b724ef
let
    theta = idata.posterior.theta[school=1:4]
    theta2 = rebuild(set(theta; school=:school2); name=:theta2)
    plot_data = Dataset(theta, theta2, idata.sample_stats.diverging)
    draw(
        data(plot_data) * mapping(
            :theta,
            :theta2 => "theta";
            col=:school,
            row=:school2,
            color=:diverging,
            markersize=:diverging => (d -> d ? 3 : 1),
        );
        figure=(; figsize=(5, 5)),
        axis=(; aspect=1),
    )
end

# ╔═╡ 7c12ac28-694c-4c12-af00-2e74bad04683
md"""
## Environment
"""

# ╔═╡ ab01574e-0b29-4325-b42e-9a360267ba95
with_terminal(Pkg.status; color=false)

# ╔═╡ e3774f8d-9885-4d49-8503-f3fc94b8b113
with_terminal(versioninfo)

# ╔═╡ Cell order:
# ╟─70f66296-349c-48a5-9b83-cc5c6cdbd514
# ╠═b1784f52-0cf2-11ed-32ba-69a8f63b48a9
# ╟─47c7a1fd-0862-49d5-879d-d59d321ce014
# ╠═34ee7907-49e7-4e32-9bd5-1c5d490a13d8
# ╟─7bdff7ba-408c-4893-9857-7cf328301a34
# ╠═cbd81558-e355-4e7a-ba51-0ba9299cb558
# ╠═898d83d1-cd3a-47a4-84b1-f2cf0f7bf959
# ╟─da4e989e-a08c-45af-8e55-20ac3ed58745
# ╠═52ef5ef5-2ea8-4e1a-8cd5-ba3aa5da6d9b
# ╟─e77c2bc3-8c23-446d-9ec4-3a155aea23e9
# ╠═0d625d42-726d-4cb3-8456-c54e1001df6d
# ╟─9b1a7256-c875-4266-b2fb-47c29f2b13c9
# ╠═0f4483b1-5820-45e2-8812-14550bff69e2
# ╟─ee4ab468-dbf6-46af-a7c5-44b82c031c2c
# ╠═9b590c11-b15a-4199-8d9f-da8e1735fed2
# ╟─ca18a574-a5c6-4d83-a5d6-b5687f569522
# ╠═746785a2-c472-467f-973a-d2390ec3e0bb
# ╟─e221c4b1-2256-4f97-bfd8-aeaf02fedc1b
# ╠═663e5edb-a751-40e7-96ec-495685453515
# ╟─6e10a67d-045a-4fa9-a7ad-90fa79015ea8
# ╠═9d8a23e1-4961-44c6-b272-afb8883df1d7
# ╟─7068ad2c-2260-4afc-b85d-229b2be6c207
# ╠═c860bb8d-c15c-4ac9-b2b7-143f6f12bf0b
# ╟─6017222f-769b-4db7-9918-af70c792a158
# ╠═b072dd93-3076-4630-bb2a-b9d490045a78
# ╟─82229ba7-101d-4fbc-96a9-ca56d899ebd5
# ╠═2a47f53a-a054-426d-b536-ccffcf62dd15
# ╟─3c939c5a-90c6-4367-9a9b-2525796425ce
# ╠═b3c8eed9-4083-457e-89dd-71d678b724ef
# ╟─7c12ac28-694c-4c12-af00-2e74bad04683
# ╠═e5e15f39-027c-42aa-a765-588ec1cd0b63
# ╠═41b1ddca-20dc-4821-9cec-d7192ef34795
# ╠═ab01574e-0b29-4325-b42e-9a360267ba95
# ╠═e3774f8d-9885-4d49-8503-f3fc94b8b113
