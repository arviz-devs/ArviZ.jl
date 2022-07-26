# [Dataset](@id dataset-api)

```@index
Pages = ["dataset.md"]
```

## Type definition

```@docs
Dataset
```

## General conversion

```@docs
convert_to_dataset
namedtuple_to_dataset
```

## DimensionalData

As a `DimensionalData.AbstractDimStack`, `Dataset` also implements the `AbstractDimStack` API and can be used like a `DimStack`.
See [DimensionalData's documentation](https://rafaqz.github.io/DimensionalData.jl/stable/) for example usage.

## Tables inteface

`Dataset` implements the [Tables](https://github.com/JuliaData/Tables.jl) interface.
This allows `Dataset`s to be used as sources for any function that can accept a table.
For example, it's straightforward to:
- write to CSV with CSV.jl
- flatten to a DataFrame with DataFrames.jl
- plot with StatsPlots.jl
- plot with AlgebraOfGraphics.jl
