# [Working with `InferenceData`](@id working-with-inference-data)

```@example wwid
using ArviZ, ArviZExampleData, DimensionalData, Statistics
```

Here we present a collection of common manipulations you can use while working with [`InferenceData`](@ref).

Let's load one of ArviZ's example datasets.
`posterior`, `posterior_predictive`, etc are the groups stored in `idata`, and they are stored as [`Dataset`](@ref)s.
In this HTML view, you can click a group name to expand a summary of the group.

```@example wwid
idata = load_example_data("centered_eight")
```

!!! info
	  `Dataset`s are [`DimensionalData.AbstractDimStack`](https://rafaqz.github.io/DimensionalData.jl/stable/api/#DimensionalData.AbstractDimStack)s and can be used identically.
	  The variables a `Dataset` contains are called "layers", and dimensions of the same name that appear in more than one layer within a `Dataset` must have the same indices.

`InferenceData` behaves like a `NamedTuple` and can be used similarly.
Note that unlike a `NamedTuple`, the groups always appear in a specific order.

```@example wwid
length(idata) # number of groups
```

```@example wwid
keys(idata) # group names
```

## Get the dataset corresponding to a single group

Group datasets can be accessed both as properties or as indexed items.

```@example wwid
post = idata.posterior
```

`post` is the dataset itself, so this is a non-allocating operation.

```@example wwid
idata[:posterior] === post
```

`InferenceData` supports a more advanced indexing syntax, which we'll see later.

## Getting a new `InferenceData` with a subset of groups

We can index by a collection of group names to get a new `InferenceData` with just those groups.
This is also non-allocating.

```@example wwid
idata_sub = idata[(:posterior, :posterior_predictive)]
```

## Adding groups to an `InferenceData`

`InferenceData` is immutable, so to add or replace groups we use [`merge`](@ref) to create a new object.

```@example wwid
merge(idata_sub, idata[(:observed_data, :prior)])
```

We can also use [`Base.setindex`](@ref) to out-of-place add or replace a single group.

```@example wwid
Base.setindex(idata_sub, idata.prior, :prior)
```

## Add a new variable

`Dataset` is also immutable.
So while the values within the underlying data arrays can be mutated, layers cannot be added or removed from `Dataset`s, and groups cannot be added/removed from `InferenceData`.

Instead, we do this out-of-place also using `merge`.

```@example wwid
merge(post, (log_tau=log.(post[:tau]),))
```

## Obtain an array for a given parameter

Let’s say we want to get the values for `mu` as an array.
Parameters can be accessed with either property or index syntax.

```@example wwid
post.tau
```

```@example wwid
post[:tau] === post.tau
```

To remove the dimensions, just use `parent` to retrieve the underlying array.

```@example wwid
parent(post.tau)
```

## Get the dimension lengths

Let’s check how many groups are in our hierarchical model.

```@example wwid
size(idata.observed_data, :school)
```

## Get coordinate/index values

What are the names of the groups in our hierarchical model?
You can access them from the coordinate name `school` in this case.

```@example wwid
DimensionalData.index(idata.observed_data, :school)
```

## Get a subset of chains

Let’s keep only chain 0 here.
For the subset to take effect on all relevant `InferenceData` groups -- `posterior`, `sample_stats`, `log_likelihood`, and `posterior_predictive` -- we will index `InferenceData` instead of `Dataset`.

Here we use DimensionalData's `At` selector.
Its [other selectors](https://rafaqz.github.io/DimensionalData.jl/stable/api/#Selectors) are also supported.

```@example wwid
idata[chain=At(0)]
```

Note that in this case, `prior` only has a chain of 0.
If it also had the other chains, we could have passed `chain=At([0, 2])` to subset by chains 0 and 2.

!!! warning
	  If we used `idata[chain=[0, 2]]` without the `At` selector, this is equivalent to `idata[chain=DimensionalData.index(idata.posterior, :chain)[0, 2]]`, that is, `[0, 2]` indexes an array of dimension indices, which here would error.
	  But if we had requested `idata[chain=[1, 2]]` we would not hit an error, but we would index the wrong chains.
	  So it's important to always use a selector to index by values of dimension indices.

## Remove the first $n$ draws (burn-in)

Let’s say we want to remove the first 100 draws from all the chains and all `InferenceData` groups with draws.
To do this we use the `..` syntax from IntervalSets.jl, which is exported by DimensionalData.

```@example wwid
idata[draw=100 .. Inf]
```

If you check the object you will see that the groups `posterior`, `posterior_predictive`, `prior`, and `sample_stats` have 400 draws compared to `idata`, which has 500.
The group `observed_data` has not been affected because it does not have the `draw` dimension.

Alternatively, you can change a subset of groups by combining indexing styles with `merge`.
Here we use this to build a new `InferenceData` where we have discarded the first 100 draws only from `posterior`.

```@example wwid
merge(idata, idata[(:posterior,), draw=100 .. Inf])
```

## Compute posterior mean values along draw and chain dimensions

To compute the mean value of the posterior samples, do the following:

```@example wwid
mean(post)
```

This computes the mean along all dimensions, discarding all dimensions and returning the result as a `NamedTuple`.
This may be what you wanted for `mu` and `tau`, which have only two dimensions (`chain` and `draw`), but maybe not what you expected for `theta`, which has one more dimension `school`.

You can specify along which dimension you want to compute the mean (or other functions), which instead returns a `Dataset`.

```@example wwid
mean(post; dims=(:chain, :draw))
```

The singleton dimensions of `chain` and `draw` now contain meaningless indices, so you may want to discard them, which you can do with `dropdims`.

```@example wwid
dropdims(mean(post; dims=(:chain, :draw)); dims=(:chain, :draw))
```

## Renaming a dimension

We can rename a dimension in a `Dataset` using DimensionalData's [`set`](https://rafaqz.github.io/DimensionalData.jl/stable/api/#DimensionalData.Dimensions.LookupArrays.set) method:

```@example wwid
theta_bis = set(post.theta; school=:school_bis)
```

We can use this, for example, to broadcast functions across multiple arrays, automatically matching up shared dimensions, using `DimensionalData.broadcast_dims`.

```@example wwid
theta_school_diff = broadcast_dims(-, post.theta, theta_bis)
```

## Compute and store posterior pushforward quantities

We use “posterior pushfoward quantities” to refer to quantities that are not variables in the posterior but deterministic computations using posterior variables.

You can compute these pushforward operations and store them as a new variable in a copy of the posterior group.

Here we'll create a new `InferenceData` with `theta_school_diff` in the posterior:

```@example wwid
idata_new = Base.setindex(idata, merge(post, (; theta_school_diff)), :posterior)
```

Once you have these pushforward quantities in an `InferenceData`, you’ll then be able to plot them with ArviZ functions, calculate stats and diagnostics on them, or save and share the `InferenceData` object with the pushforward quantities included.

Here we compute the [`mcse`](@ref) of `theta_school_diff`:

```@example wwid
mcse(idata_new.posterior).theta_school_diff
```

## Advanced subsetting

To select the value corresponding to the difference between the Choate and Deerfield schools do:

```@example wwid
school_idx = ["Choate", "Hotchkiss", "Mt. Hermon"]
school_bis_idx = ["Deerfield", "Choate", "Lawrenceville"]
theta_school_diff[school=At(school_idx), school_bis=At(school_bis_idx)]
```

## Add new chains using `cat`

Suppose after checking the `mcse` and realizing you need more samples, you rerun the model with two chains and obtain an `idata_rerun` object.

```@example wwid
idata_rerun = InferenceData(; posterior=set(post[chain=At([0, 1])]; chain=[4, 5]))
```

You can combine the two using [`cat`](@ref).

```@example wwid
cat(idata[[:posterior]], idata_rerun; dims=:chain)
```
