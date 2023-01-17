# A trust-region type and basic utility functions.
import NLPModels: reset!
export TrustRegionException, acceptable, aredpred!, reset!, update!

"Exception type raised in case of error."
mutable struct TrustRegionException <: Exception
  msg::String
end

"""`AbstractTrustRegion`

An abstract trust region type so that specific trust regions define update
rules differently. Child types must have at least the following real fields:

- `acceptance_threshold`
- `initial_radius`
- `radius`
- `ratio`

and the following function:

- `update!(tr, step_norm)`
"""
abstract type AbstractTrustRegion{T, V} end

"""
    ared, pred, good_grad = aredpred_common(nlp, f, f_trial, Δm, x_trial, step, g_trial, slope)

Compute the actual and predicted reductions `∆f` and `Δm`, where
`ared = Δf = f_trial - f` is the actual reduction is an objective/merit/penalty function,
`pred = Δm = m_trial - m` is the reduction predicted by the model `m` of `f`.
We assume that `m` is being minimized, and therefore that `Δm < 0`.

`good_grad` is `true` if the gradient of `nlp` at `x_trial` has been updated and stored in `g_trial`.
"""
function aredpred_common(
  nlp::AbstractNLPModel{T, V},
  f::T,
  f_trial::T,
  Δm::T,
  x_trial::V,
  step::V,
  g_trial::V,
  slope::T,
) where {T, V}
  absf = abs(f)
  ϵ = eps(T)
  pred = Δm - max(one(T), absf) * 10 * ϵ

  good_grad = false
  ared = f_trial - f + max(one(T), absf) * 10 * ϵ
  if (abs(Δm) < 10_000 * ϵ) || (abs(ared) < 10_000 * ϵ * absf)
    # correct for roundoff error
    grad!(nlp, x_trial, g_trial)
    good_grad = true
    slope_trial = dot(g_trial, step)
    ared = (slope_trial + slope) / 2
  end
  return ared, pred, good_grad
end

"""`ared, pred = aredpred(tr, nlp, f, f_trial, Δm, x_trial, step, slope)`

Compute the actual and predicted reductions `∆f` and `Δm`, where
`ared = Δf = f_trial - f` is the actual reduction is an objective/merit/penalty function,
`pred = Δm = m_trial - m` is the reduction predicted by the model `m` of `f`.
We assume that `m` is being minimized, and therefore that `Δm < 0`.

If `tr.good_grad` is `true`, then the gradient of `nlp` at `x_trial` is stored in `tr.gt`.
"""
function aredpred!(
  tr::AbstractTrustRegion{T, V},
  nlp::AbstractNLPModel{T, V},
  f::T,
  f_trial::T,
  Δm::T,
  x_trial::V,
  step::V,
  slope::T,
) where {T, V}
  ared, pred, tr.good_grad = aredpred_common(nlp, f, f_trial, Δm, x_trial, step, tr.gt, slope)
  return ared, pred
end

"""`acceptable(tr)`

Return `true` if a step is acceptable, i.e. large or equal to `tr.acceptance_threshold`.
"""
function acceptable(tr::AbstractTrustRegion)
  return tr.ratio >= tr.acceptance_threshold
end

"""`reset!(tr)`

Reset the trust-region radius to its initial value.
"""
function reset!(tr::AbstractTrustRegion)
  tr.radius = tr.initial_radius
  return tr
end

"""`update!(tr, step_norm)`

Update the trust-region radius based on the ratio `tr.ratio` of actual vs. predicted reduction
and the step norm.
"""
function update! end

include("basic-trust-region.jl")
include("tron-trust-region.jl")
