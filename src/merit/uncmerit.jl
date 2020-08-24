export UncMerit

@doc raw"""
  UncMerit(nlp; kwargs...)
  UncMerit(nlp, η; kwargs...) # η is ignored

Creates a merit wrapper for unconstrained or bound-countrained problems.
Formally accepts the same constructor as other merit functions, but in practice it ignores any constrained-related value, even if `nlp` is constrained.
"""
mutable struct UncMerit{M <: AbstractNLPModel, T <: Real, V <: AbstractVector} <: AbstractMeritModel{M,T,V}
  meta :: NLPModelMeta
  counters :: Counters
  nlp  :: M
  η  :: T # ignored in practice
  fx   :: T
  gx   :: V
  cx   :: V # ignored in practice
  Ad   :: V # ignored in practice
end

function UncMerit(
  nlp  :: M,
  η  :: T = 0.0;
  fx   :: T = T(Inf),
  gx   :: V = fill(T(Inf), nlp.meta.nvar),
  cx   :: V = T[],
  Ad   :: V = T[],
) where {M <: AbstractNLPModel, T <: Real, V <: AbstractVector{<: T}}
  meta = NLPModelMeta(nlp.meta.nvar)
  UncMerit{M,T,V}(meta, Counters(), nlp, η, fx, gx, cx, Ad)
end

function dualobj(merit :: UncMerit)
  merit.fx
end

function primalobj(merit :: UncMerit)
  zero(eltype(merit.η))
end

function derivative(merit :: UncMerit, x :: AbstractVector, d :: AbstractVector; update :: Bool = true)
  @lencheck merit.meta.nvar x d
  if update
    grad!(merit.nlp, x, merit.gx)
  end
  return dot(merit.gx, d)
end