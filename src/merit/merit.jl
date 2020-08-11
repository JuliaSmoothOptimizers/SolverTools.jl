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
abstract type AbstractMeritModel{M,T,V} <: AbstractNLPModel end

"""
    obj(merit, x; update=true)

Evaluates the `merit` model at `x`.
This will call `obj` and `cons!` to update the internal values of `fx` and `cx`, unless `update=false`.
"""
function NLPModels.obj(merit :: AbstractMeritModel, x :: AbstractVector; update :: Bool=false)
  throw(MethorError(obj, (merit, x)))
end

"""
    derivative(merit, x, d; update=true)

Computes the directional derivative of `merit` at `x` on direction `d`.
This will call `grad!` and `jprod` to update the internal values of `gx` and `Ad`, unless `update=false`.
This function assumes that `cx` is correct, though.
"""
function derivative end

include("auglagmerit.jl")
include("l1merit.jl")
include("uncmerit.jl")