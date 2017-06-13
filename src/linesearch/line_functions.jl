importall NLPModels

export LineFunction
export obj, grad, derivative, grad!, derivative!, hess

"""A type to represent the restriction of a function to a direction.
Given f : R → Rⁿ, x ∈ Rⁿ and a nonzero direction d ∈ Rⁿ,

    ϕ = LineFunction(nlp, x, d)

represents the function ϕ : R → R defined by

    ϕ(t) := f(x + td).
"""
type LineFunction <: AbstractNLPModel
  nlp :: AbstractNLPModel
  x :: Vector
  d :: Vector
end

"""`obj(f, t)` evaluates the objective of the `LineFunction`

    ϕ(t) := f(x + td).
"""
obj(f :: LineFunction, t :: Float64) = obj(f.nlp, f.x + t * f.d)


"""`grad(f, t)` evaluates the first derivative of the `LineFunction`

    ϕ(t) := f(x + td),

i.e.,

    ϕ'(t) = ∇f(x + td)ᵀd.
"""
grad(f :: LineFunction, t :: Float64) = dot(grad(f.nlp, f.x + t * f.d), f.d)
derivative(f :: LineFunction, t :: Float64) = grad(f, t)

"""`grad!(f, t, g)` evaluates the first derivative of the `LineFunction`

    ϕ(t) := f(x + td),

i.e.,

    ϕ'(t) = ∇f(x + td)ᵀd.

The gradient ∇f(x + td) is stored in `g`.
"""
grad!(f :: LineFunction, t :: Float64, g :: Vector{Float64}) = dot(grad!(f.nlp, f.x + t * f.d, g), f.d)
derivative!(f :: LineFunction, t :: Float64, g :: Vector{Float64}) = grad!(f, t, g)

"""Evaluate the second derivative of the `LineFunction`

    ϕ(t) := f(x + td),

i.e.,

    ϕ"(t) = dᵀ∇²f(x + td)d.
"""
hess(f :: LineFunction, t :: Float64) = dot(f.d, hprod(f.nlp, f.x + t * f.d, f.d))
