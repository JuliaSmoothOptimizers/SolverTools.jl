export C1LineFunction, C2LineFunction
export obj, grad, grad!, hess

# Import methods that we extend.
import NLPModels.obj, NLPModels.grad, NLPModels.grad!, NLPModels.hess


"""A type to represent the restriction of a C1 function to a direction.
Given f : R → Rⁿ, x ∈ Rⁿ and a nonzero direction d ∈ Rⁿ,

    ϕ = C1LineFunction(f, x, d)

represents the function ϕ : R → R defined by

    ϕ(t) := f(x + td).
"""
type C1LineFunction
  nlp :: AbstractNLPModel
  x :: Vector
  d :: Vector
end


"""`obj(f, t)` evaluates the objective of the `C1LineFunction`

    ϕ(t) := f(x + td).
"""
obj(f :: C1LineFunction, t :: Float64) = obj(f.nlp, f.x + t * f.d)


"""`grad(f, t)` evaluates the first derivative of the `C1LineFunction`

    ϕ(t) := f(x + td),

i.e.,

    ϕ'(t) = ∇f(x + td)ᵀd.
"""
grad(f :: C1LineFunction, t :: Float64) = dot(grad(f.nlp, f.x + t * f.d), f.d)


"""`grad!(f, t, g)` evaluates the first derivative of the `C1LineFunction`

    ϕ(t) := f(x + td),

i.e.,

    ϕ'(t) = ∇f(x + td)ᵀd.

The gradient ∇f(x + td) is stored in `g`.
"""
grad!(f :: C1LineFunction, t :: Float64, g :: Vector{Float64}) = dot(grad!(f.nlp, f.x + t * f.d, g), f.d)


"""A type to represent the restriction of a C2 function to a direction.
Given f : R → Rⁿ, x ∈ Rⁿ and a nonzero direction d ∈ Rⁿ,

    ϕ = C2LineFunction(f, x, d)

represents the function ϕ : R → R defined by

    ϕ(t) := f(x + td).
"""
type C2LineFunction
  nlp :: AbstractNLPModel
  x :: Vector
  d :: Vector
end


"""`obj(f, t)` evaluates the objective of the `C2LineFunction`

    ϕ(t) := f(x + td).
"""
obj(f :: C2LineFunction, t :: Float64) = obj(f.nlp, f.x + t * f.d)


"""`grad(f, t)` evaluates the first derivative of the `C2LineFunction`

    ϕ(t) := f(x + td),

i.e.,

    ϕ'(t) = ∇f(x + td)ᵀd.
"""
grad(f :: C2LineFunction, t :: Float64) = dot(grad(f.nlp, f.x + t * f.d), f.d)


"""`grad!(f, t, g)` evaluates the first derivative of the `C2LineFunction`

    ϕ(t) := f(x + td),

i.e.,

    ϕ'(t) = ∇f(x + td)ᵀd.

The gradient ∇f(x + td) is stored in `g`.
"""
grad!(f :: C2LineFunction, t :: Float64, g :: Vector{Float64}) = dot(grad!(f.nlp, f.x + t * f.d, g), f.d)


"""Evaluate the second derivative of the `C2LineFunction`

    ϕ(t) := f(x + td),

i.e.,

    ϕ"(t) = dᵀ∇²f(x + td)d.
"""
hess(f :: C2LineFunction, t :: Float64) = dot(f.d, hprod(f.nlp, f.x + t * f.d, f.d))
