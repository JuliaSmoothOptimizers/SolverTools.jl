import NLPModels.obj
export AbstractMeritModel, obj, derivative

"""
    AbstractMeritModel

Model for merit functions. All models should store
- `nlp`: The NLP with the corresponding problem.
- `fx`: The objective at some point.
- `cx`: The constraints vector at some point.
- `gx`: The constraints vector at some point.
- `Ad`: The Jacobian-direction product.
"""
abstract type AbstractMeritModel end

"""
    obj(merit, x; update=true)

Computes the `merit` function value at `x`.
This will call `obj` and `cons` for the internal model, unless `update` is called with `false`.
The option exist to allow updating the `η` parameter without recomputing `fx` and `cx`.
"""
function obj end

"""
    derivative(merit, x, d; update=true)

Computes the derivative derivative of `merit` at `x` on direction `d`.
This will call `grad!` and `jprod` to update the internal values of `gx` and `Ad`, but will assume that `cx` is correct.
The option exist to allow updating the `η` parameter without recomputing `fx` and `cx`.
"""
function derivative! end

include("auglagmerit.jl")
include("l1merit.jl")
