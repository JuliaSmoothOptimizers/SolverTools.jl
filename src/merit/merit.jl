using NLPModels
export AbstractMeritModel, obj, derivative

"""
    AbstractMeritModel

Model for merit functions. All models should store
- `nlp`: The NLP with the corresponding problem.
- `η`: The merit parameter.
- `fx`: The objective at some point.
- `cx`: The constraints vector at some point.
- `gx`: The constraints vector at some point.
- `Ad`: The Jacobian-direction product.

All models allow a constructor of form

    Merit(nlp, η; fx=Inf, cx=[Inf,…,Inf], gx=[Inf,…,Inf], Ad=[Inf,…,Inf])

Additional arguments and constructors may be provided.

An AbstractMeritModel is an AbstractNLPModel, but the API may not be completely implemented. For
instance, the `L1Merit` model doesn't provide any gradient function, but it provides a directional
derivative function.

Furthermore, all implemented methods accept an `update` keyword that defaults to `true`. It is used
to determine whether the internal stored values should be updated or not.
"""
abstract type AbstractMeritModel <: AbstractNLPModel end

"""
    derivative(merit, x, d; update=true)

Computes the directional derivative of `merit` at `x` on direction `d`.
This will call `grad!` and `jprod` to update the internal values of `gx` and `Ad`, but will assume that `cx` is correct.
The option exist to allow updating the `η` parameter without recomputing `fx` and `cx`.
"""
function derivative end

include("auglagmerit.jl")
include("l1merit.jl")
