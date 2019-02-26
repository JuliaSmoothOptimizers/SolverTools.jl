# A trust-region type and basic utility functions.
import NLPModels: reset!
export TrustRegionException, acceptable, aredpred, get_property, ratio, ratio!, reset!, set_property!, update!

"Exception type raised in case of error."
mutable struct TrustRegionException <: Exception
  msg  :: String
end

"""`AbstractTrustRegion`

An abstract trust region type so that specific trust regions define update
rules differently. Child types must have at least the following fields:

- `acceptance_threshold :: AbstractFloat`
- `initial_radius :: AbstractFloat`
- `radius :: AbstractFloat`
- `ratio :: AbstractFloat`

and the following function:

- `update!(tr, step_norm)`
"""
abstract type AbstractTrustRegion end

"""`ared, pred = aredpred(nlp, f, f_trial, Δm, x_trial, step, slope)`

Compute the actual and predicted reductions `∆f` and `Δm`, where
`Δf = f_trial - f` is the actual reduction is an objective/merit/penalty function,
`Δm = m_trial - m` is the reduction predicted by the model `m` of `f`.
We assume that `m` is being minimized, and therefore that `Δm < 0`.
"""
function aredpred(nlp :: AbstractNLPModel, f :: T, f_trial :: T, Δm :: T,
                  x_trial :: Vector{T}, step :: Vector{T}, slope :: T) where T <: AbstractFloat
  absf = abs(f)
  ϵ = eps(T)
  pred = Δm - max(one(T), absf) * 10 * ϵ

  ared = f_trial - f + max(one(T), absf) * 10 * ϵ
  if (abs(Δm) < 10_000 * ϵ) || (abs(ared) < 10_000 * ϵ * absf)
    # correct for roundoff error
    g_trial = grad(nlp, x_trial)
    slope_trial = dot(g_trial, step)
    ared = (slope_trial + slope) / 2
  end
  return ared, pred
end


"""`acceptable(tr)`

Return `true` if a step is acceptable
"""
function acceptable(tr :: AbstractTrustRegion)
  return tr.ratio >= tr.acceptance_threshold
end

"""`reset!(tr)`

Reset the trust-region radius to its initial value
"""
function reset!(tr :: AbstractTrustRegion)
  tr.radius = tr.initial_radius
  return tr
end


"""A basic getter for `AbstractTrustRegion` instances.
Should be overhauled when it's possible to overload `getfield()`
and `setfield!()`. See
https://github.com/JuliaLang/julia/issues/1974
"""
function get_property(tr :: AbstractTrustRegion, prop :: Symbol)
  # All fields are gettable.
  gettable = fieldnames(typeof(tr))
  prop in gettable || throw(TrustRegionException("Unknown property: $prop"))
  getfield(tr, prop)
end

"""A basic setter for `AbstractTrustRegion` instances.
"""
function set_property!(tr :: AbstractTrustRegion, prop :: Symbol, value :: Any)
  gettable = fieldnames(typeof(tr))
  prop in gettable || throw(TrustRegionException("Unknown property: $prop"))
  setfield!(tr, prop, value)
end

"""`update!(tr, step_norm)`

Update the trust-region radius based on the ratio of actual vs. predicted reduction
and the step norm.
"""
function update!(tr :: AbstractTrustRegion, ratio :: AbstractFloat, step_norm :: AbstractFloat)
  throw(NotImplementedError("`update!` not implemented for this TrustRegion type"))
end

include("basic-trust-region.jl")
include("tron-trust-region.jl")
