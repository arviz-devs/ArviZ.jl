# [InferenceData](@id inferencedata-api)

```@index
Pages = ["inference_data.md"]
```

## Type definition

```@docs
InferenceData
```

## Property interface

```@docs
getproperty
propertynames
```

## Indexing interface

```@docs
getindex
Base.setindex
```

## Iteration interface

`InferenceData` also implements the same iteration interface as its underlying `NamedTuple`.
That is, iterating over an `InferenceData` iterates over its groups.

## General conversion

```@docs
convert_to_inference_data
from_dict
from_namedtuple
```

## General functions

```@docs
cat
merge
```
