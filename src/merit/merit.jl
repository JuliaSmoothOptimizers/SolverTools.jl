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
"""
abstract type AbstractMeritModel end

"""
    obj(merit, x; update=true)

Computes the `merit` function value at `x`.
This will call `obj` and `cons` for the internal model, unless `update` is called with `false`.
The option exist to allow updating the `η` parameter without recomputing `fx` and `cx`.
"""
function NLPModels.obj(merit::AbstractMeritModel, x::AbstractVector)
    # I'd prefer to use only `function NLPModels.obj end` instead, but it doesn't work and using
    # only `function obj end` overwrites the docstring
    throw(MethodError(NLPModels.obj, (merit, x)))
end

"""
    derivative(merit, x, d; update=true)

Computes the derivative derivative of `merit` at `x` on direction `d`.
This will call `grad!` and `jprod` to update the internal values of `gx` and `Ad`, but will assume that `cx` is correct.
The option exist to allow updating the `η` parameter without recomputing `fx` and `cx`.
"""
function derivative end

include("auglagmerit.jl")
include("l1merit.jl")
