# A trust-region type and basic utility functions.
import NLPModels: reset!
export TrustRegionException, acceptable, get_property, ratio, ratio!, reset!, update!

global const ϵ = eps(Float64)

"Exception type raised in case of error."
mutable struct TrustRegionException <: Exception
  msg  :: String
end

"""`AbstractTrustRegion`

An abstract trust region type so that specific trust regions define update
rules differently. Child types must have at least the following fields:

- `acceptance_threshold :: Float64`
- `initial_radius :: Float64`
- `radius :: Float64`
- `ratio :: Float64`

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
function aredpred(nlp :: AbstractNLPModel, f :: Float64, f_trial :: Float64, Δm :: Float64,
                  x_trial :: Vector{Float64}, step :: Vector{Float64}, slope :: Float64)
  absf = abs(f)
  pred = Δm - max(1.0, absf) * 10.0 * ϵ

  ared = f_trial - f + max(1.0, absf) * 10.0 * ϵ
  if (abs(Δm) < 1.0e+4 * ϵ) || (abs(ared) < 1.0e+4 * ϵ * absf)
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
function update!(tr :: AbstractTrustRegion, ratio :: Float64, step_norm :: Float64)
  throw(NotImplementedError("`update!` not implemented for this TrustRegion type"))
end

include("basic-trust-region.jl")
include("tron-trust-region.jl")
