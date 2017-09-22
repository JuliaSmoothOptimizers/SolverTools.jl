using NLPModels

"""`AugLagModel`

Augmented Lagrangian Model.
Transforms this

    min  f(x)
    s.to c(x) = 0
         ℓ ≦ x ≦ u

into

    min  f(x) + ¹/₂ρ‖c(x)‖² + λᵀc(x)
    s.to ℓ ≦ x ≦ u
"""
type AugLagModel <: AbstractNLPModel
  meta :: NLPModelMeta
  model :: AbstractNLPModel
  ρ :: Float64
  λ :: Vector
end

counters(nlp :: AugLagModel) = counters(nlp.model)

function AugLagModel(model :: AbstractNLPModel, ρ :: Float64)
  m = model.meta
  meta = NLPModelMeta(m.nvar, x0=m.x0, lvar=m.lvar, uvar=m.uvar, name=m.name)
  return AugLagModel(meta, model, ρ, m.y0)
end

function update_rho(nlp :: AugLagModel, ρ :: Float64)
  nlp.ρ = ρ
end

function update_multiplier(nlp :: AugLagModel, λ :: Vector)
  nlp.λ .= λ
end

function obj(nlp :: AugLagModel, x :: Vector)
  cx = cons(nlp.model, x)
  return obj(nlp.model, x) + nlp.ρ * dot(cx, cx) / 2 + dot(nlp.λ, cx)
end

function grad!(nlp :: AugLagModel, x :: Vector, gx :: Vector)
  cx = cons(nlp.model, x)
  grad!(nlp.model, x, gx)
  @views gx[1:nlp.meta.nvar] += jtprod(nlp.model, x, nlp.λ + nlp.ρ * cx)
  return gx
end

function grad(nlp :: AugLagModel, x :: Vector)
  gx = zeros(nlp.meta.nvar)
  grad!(nlp, x, gx)
end

function hprod!(nlp :: AugLagModel, x :: Vector, v :: Vector, Hv :: Vector; obj_weight :: Real = 1.0, y=Float64[])
  cx = cons(nlp.model, x)
  hprod!(nlp.model, x, v, Hv, y=nlp.λ + nlp.ρ * cx, obj_weight=obj_weight)
  @views Hv[1:nlp.meta.nvar] += nlp.ρ * jtprod(nlp.model, x, jprod(nlp.model, x, v))
  return Hv
end

function hprod(nlp :: AugLagModel, x :: Vector, v :: Vector; obj_weight :: Real = 1.0, y=Float64[])
  Hv = zeros(nlp.meta.nvar)
  hprod!(nlp, x, v, Hv, obj_weight=obj_weight)
end

function hess(nlp :: AugLagModel, x :: Vector; obj_weight :: Real = 1.0, y=Float64[])
  cx = cons(nlp.model, x)
  Jx = jac(nlp.model, x)
  hess(nlp.model, x, y=nlp.λ + nlp.ρ * cx, obj_weight=obj_weight) + nlp.ρ * Jx' * Jx
end

function hess_op(nlp :: AugLagModel, x :: Vector; obj_weight :: Real = 1.0, y=Float64[])
  cx = cons(nlp.model, x)
  Jx = jac_op(nlp.model, x)
  hess_op(nlp.model, x, y=nlp.λ + nlp.ρ * cx) + nlp.ρ * Jx' * Jx
end
