import NLPModels: obj, grad, grad!, objgrad!, objgrad, hess

export LineModel
export obj, grad, derivative, grad!, objgrad!, objgrad, derivative!, hess, redirect!

"""A type to represent the restriction of a function to a direction.
Given f : R → Rⁿ, x ∈ Rⁿ and a nonzero direction d ∈ Rⁿ,

    ϕ = LineModel(nlp, x, d)

represents the function ϕ : R → R defined by

    ϕ(t) := f(x + td).
"""
mutable struct LineModel <: AbstractNLPModel
  meta::NLPModelMeta
  counters::Counters
  nlp::AbstractNLPModel
  x::AbstractVector
  d::AbstractVector
end

function LineModel(nlp::AbstractNLPModel, x::AbstractVector, d::AbstractVector)
  meta = NLPModelMeta(1, x0 = zeros(eltype(x), 1), name = "LineModel to $(nlp.meta.name))")
  return LineModel(meta, Counters(), nlp, x, d)
end

"""`redirect!(ϕ, x, d)`

Change the values of x and d of the LineModel ϕ, but retains the counters.
"""
function redirect!(ϕ::LineModel, x::AbstractVector, d::AbstractVector)
  ϕ.x, ϕ.d = x, d
  return ϕ
end

"""`obj(f, t)` evaluates the objective of the `LineModel`

    ϕ(t) := f(x + td).
"""
function obj(f::LineModel, t::AbstractFloat)
  NLPModels.increment!(f, :neval_obj)
  return obj(f.nlp, f.x + t * f.d)
end

"""`grad(f, t)` evaluates the first derivative of the `LineModel`

    ϕ(t) := f(x + td),

i.e.,

    ϕ'(t) = ∇f(x + td)ᵀd.
"""
function grad(f::LineModel, t::AbstractFloat)
  NLPModels.increment!(f, :neval_grad)
  return dot(grad(f.nlp, f.x + t * f.d), f.d)
end
derivative(f::LineModel, t::AbstractFloat) = grad(f, t)

"""`grad!(f, t, g)` evaluates the first derivative of the `LineModel`

    ϕ(t) := f(x + td),

i.e.,

    ϕ'(t) = ∇f(x + td)ᵀd.

The gradient ∇f(x + td) is stored in `g`.
"""
function grad!(f::LineModel, t::AbstractFloat, g::AbstractVector)
  NLPModels.increment!(f, :neval_grad)
  return dot(grad!(f.nlp, f.x + t * f.d, g), f.d)
end
derivative!(f::LineModel, t::AbstractFloat, g::AbstractVector) = grad!(f, t, g)

"""`objgrad!(f, t, g)` evaluates the objective and first derivative of the `LineModel`

    ϕ(t) := f(x + td),

and

    ϕ'(t) = ∇f(x + td)ᵀd.

The gradient ∇f(x + td) is stored in `g`.
"""
function objgrad!(f::LineModel, t::AbstractFloat, g::AbstractVector)
  NLPModels.increment!(f, :neval_obj)
  NLPModels.increment!(f, :neval_grad)
  fx, gx = objgrad!(f.nlp, f.x + t * f.d, g)
  return fx, dot(gx, f.d)
end

"""`objgrad(f, t)` evaluates the objective and first derivative of the `LineModel`

    ϕ(t) := f(x + td),

and

    ϕ'(t) = ∇f(x + td)ᵀd.
"""
function objgrad(f::LineModel, t::AbstractFloat)
  return obj(f, t), grad(f, t)
end

"""Evaluate the second derivative of the `LineModel`

    ϕ(t) := f(x + td),

i.e.,

    ϕ"(t) = dᵀ∇²f(x + td)d.
"""
function hess(f::LineModel, t::AbstractFloat)
  NLPModels.increment!(f, :neval_hess)
  return dot(f.d, hprod(f.nlp, f.x + t * f.d, f.d))
end
