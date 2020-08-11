export AugLagMerit

@doc raw"""
  AugLagMerit(nlp, η; kwargs...)

Creates an augmented Lagrangian merit function for the equality constrained problem
```math
\min f(x) \quad \text{s.to} \quad c(x) = 0
```
defined by
```math
\phi(x, yₖ; η) = f(x) + yₖᵀc(x) + η ½\|c(x)\|²
```

In addition to the keyword arguments declared in [`AbstractMeritModel`](@ref), an `AugLagMerit` also
accepts the argument `y`.
"""
mutable struct AugLagMerit{M <: AbstractNLPModel, T <: Real, V <: AbstractVector} <: AbstractMeritModel{M,T,V}
  meta :: NLPModelMeta
  counters :: Counters
  nlp  :: M
  η  :: T
  fx   :: T
  gx   :: V
  cx   :: V
  Ad   :: V
  y  :: V
  y⁺   :: V
  Jᵀy⁺ :: V
  Jv   :: V
  JᵀJv   :: V
end

function AugLagMerit(
  nlp  :: M,
  η  :: T;
  fx   :: T = T(Inf),
  gx   :: V = fill(T(Inf), nlp.meta.nvar),
  cx   :: V = fill(T(Inf), nlp.meta.ncon),
  Ad   :: V = fill(T(Inf), nlp.meta.ncon),
  y  :: V = fill(T(Inf), nlp.meta.ncon),
  y⁺   :: V = fill(T(Inf), nlp.meta.ncon), # y + η * c(x)
  Jᵀy⁺ :: V = fill(T(Inf), nlp.meta.nvar),
  Jv   :: V = fill(T(Inf), nlp.meta.ncon),
  JᵀJv :: V = fill(T(Inf), nlp.meta.nvar)
) where {M <: AbstractNLPModel, T <: Real, V <: AbstractVector{<: T}}
  meta = NLPModelMeta(nlp.meta.nvar)
  AugLagMerit{M,T,V}(meta, Counters(), nlp, η, fx, gx, cx, Ad, y, y⁺, Jᵀy⁺, Jv, JᵀJv)
end

function NLPModels.obj(merit :: AugLagMerit, x :: AbstractVector; update :: Bool = true)
  @lencheck merit.meta.nvar x
  NLPModels.increment!(merit, :neval_obj)
  if update
    merit.fx = obj(merit.nlp, x)
    merit.nlp.meta.ncon > 0 && cons!(merit.nlp, x, merit.cx)
  end
  return merit.fx + dot(merit.y, merit.cx) + merit.η * dot(merit.cx, merit.cx) / 2
end

function derivative(merit :: AugLagMerit, x :: AbstractVector, d :: AbstractVector; update :: Bool = true)
  @lencheck merit.meta.nvar x d
  if update
    grad!(merit.nlp, x, merit.gx)
    merit.nlp.meta.ncon > 0 && jprod!(merit.nlp, x, d, merit.Ad)
  end
  if merit.nlp.meta.ncon == 0
    return dot(merit.gx, d)
  else
    return dot(merit.gx, d) + dot(merit.y, merit.Ad) + merit.η * dot(merit.cx, merit.Ad)
  end
end

function NLPModels.grad!(merit :: AugLagMerit, x :: AbstractVector, g :: AbstractVector; update :: Bool = true)
  @lencheck merit.meta.nvar x g
  NLPModels.increment!(merit, :neval_grad)
  if update
    grad!(nlp.model, x, merit.gx)
    merit.nlp.meta.ncon > 0 && cons!(merit.nlp, x, merit.cx)
    merit.y⁺ .= merit.y .+ merit.η .* merit.cx
    merit.nlp.meta.ncon > 0 && jtprod!(merit.nlp, x, merit.y⁺, merit.Jᵀy⁺)
  end
  g .= merit.gx .+ merit.Jᵀy⁺
  return g
end

function NLPModels.objgrad!(merit :: AugLagMerit, x :: AbstractVector, g :: AbstractVector)
  @lencheck merit.meta.nvar x g
  NLPModels.increment!(merit, :neval_obj)
  NLPModels.increment!(merit, :neval_grad)
  if update
    merit.fx = obj(merit.nlp, x)
    grad!(nlp.model, x, merit.gx)
    merit.nlp.meta.ncon > 0 && cons!(merit.nlp, x, merit.cx)
    merit.y⁺ .= merit.y .+ merit.η .* merit.cx
    merit.nlp.meta.ncon > 0 && jtprod!(merit.nlp, x, merit.y⁺, merit.Jᵀy⁺)
  end
  f = merit.fx + dot(merit.y, merit.cx) + merit.η * dot(merit.cx, merit.cx) / 2
  g .= merit.gx .+ merit.Jᵀy⁺
  return f, g
end

function NLPModels.hprod!(merit :: AugLagMerit, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Float64 = 1.0)
  @lencheck merit.meta.nvar x v Hv
  NLPModels.increment!(merit, :neval_hprod)
  if update
    merit.nlp.meta.ncon > 0 && cons!(merit.nlp, x, merit.cx)
    merit.y⁺ .= merit.y .+ merit.η .* merit.cx
  end
  jprod!(merit.model, x, v, merit.Jv)
  jtprod!(merit.model, x, merit.Jv, merit.JᵀJv)
  hprod!(merit.model, x, merit.y⁺, v, Hv, obj_weight = obj_weight)
  Hv .+= merit.η * merit.JᵀJv
  return Hv
end