importall NLPModels

export LineModel
export obj, grad, derivative, grad!, derivative!, hess

"""A type to represent the restriction of a function to a direction.
Given f : R → Rⁿ, x ∈ Rⁿ and a nonzero direction d ∈ Rⁿ,

    ϕ = LineModel(nlp, x, d)

represents the function ϕ : R → R defined by

    ϕ(t) := f(x + td).
"""
type LineModel <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
  nlp :: AbstractNLPModel
  x :: Vector
  d :: Vector
end

function LineModel(nlp :: AbstractNLPModel, x :: Vector{Float64}, d :: Vector{Float64})
  meta = NLPModelMeta(1, x0=zeros(1), name="LineModel to $(nlp.meta.name))")
  return LineModel(meta, Counters(), nlp, x, d)
end

"""`redirect!(ϕ, x, d)`

Change the values of x and d of the LineModel ϕ, but retains the counters.
"""
function redirect!(ϕ :: LineModel, x :: Vector{Float64}, d :: Vector{Float64})
  ϕ.x, ϕ.d = x, d
  return ϕ
end

"""`obj(f, t)` evaluates the objective of the `LineModel`

    ϕ(t) := f(x + td).
"""
obj(f :: LineModel, t :: Float64) = obj(f.nlp, f.x + t * f.d)


"""`grad(f, t)` evaluates the first derivative of the `LineModel`

    ϕ(t) := f(x + td),

i.e.,

    ϕ'(t) = ∇f(x + td)ᵀd.
"""
grad(f :: LineModel, t :: Float64) = dot(grad(f.nlp, f.x + t * f.d), f.d)
derivative(f :: LineModel, t :: Float64) = grad(f, t)

"""`grad!(f, t, g)` evaluates the first derivative of the `LineModel`

    ϕ(t) := f(x + td),

i.e.,

    ϕ'(t) = ∇f(x + td)ᵀd.

The gradient ∇f(x + td) is stored in `g`.
"""
grad!(f :: LineModel, t :: Float64, g :: Vector{Float64}) = dot(grad!(f.nlp, f.x + t * f.d, g), f.d)
derivative!(f :: LineModel, t :: Float64, g :: Vector{Float64}) = grad!(f, t, g)

"""Evaluate the second derivative of the `LineModel`

    ϕ(t) := f(x + td),

i.e.,

    ϕ"(t) = dᵀ∇²f(x + td)d.
"""
hess(f :: LineModel, t :: Float64) = dot(f.d, hprod(f.nlp, f.x + t * f.d, f.d))
