import NLPModels: obj, grad, grad!, objgrad!, objgrad, hess

export LineModel
export obj, grad, derivative, grad!, objgrad!, objgrad, derivative!, hess, hess!, redirect!

"""A type to represent the restriction of a function to a direction.
Given f : R → Rⁿ, x ∈ Rⁿ and a nonzero direction d ∈ Rⁿ,

    ϕ = LineModel(nlp, x, d)

represents the function ϕ : R → R defined by

    ϕ(t) := f(x + td).
"""
mutable struct LineModel{T, S, M <: AbstractNLPModel{T, S}} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  nlp::M
  x::S
  d::S
  xt::S
end

function LineModel(nlp::AbstractNLPModel{T, S}, x::S, d::S; xt::S = similar(x)) where {T, S}
  meta = NLPModelMeta{T, S}(1, x0 = similar(x, 1), name = "LineModel to $(nlp.meta.name))")
  return LineModel(meta, Counters(), nlp, x, d, xt)
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
  @. f.xt = f.x + t * f.d
  return obj(f.nlp, f.xt)
end

"""`grad(f, t)` evaluates the first derivative of the `LineModel`

    ϕ(t) := f(x + td),

i.e.,

    ϕ'(t) = ∇f(x + td)ᵀd.
"""
function grad(f::LineModel, t::AbstractFloat)
  NLPModels.increment!(f, :neval_grad)
  @. f.xt = f.x + t * f.d
  return dot(grad(f.nlp, f.xt), f.d)
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
  @. f.xt = f.x + t * f.d
  return dot(grad!(f.nlp, f.xt, g), f.d)
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
  @. f.xt = f.x + t * f.d
  fx, _ = objgrad!(f.nlp, f.xt, g)
  return fx, dot(g, f.d)
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
  @. f.xt = f.x + t * f.d
  return dot(f.d, hprod(f.nlp, f.xt, f.d))
end

"""Evaluate the second derivative of the `LineModel`

    ϕ(t) := f(x + td),

i.e.,

    ϕ"(t) = dᵀ∇²f(x + td)d.
"""
function hess!(f::LineModel, t::AbstractFloat, Hv::AbstractVector)
  NLPModels.increment!(f, :neval_hess)
  @. f.xt = f.x + t * f.d
  return dot(f.d, hprod!(f.nlp, f.xt, f.d, Hv))
end
