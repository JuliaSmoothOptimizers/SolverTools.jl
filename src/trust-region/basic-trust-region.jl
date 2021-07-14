export TrustRegion

"""Basic trust region type."""
mutable struct TrustRegion{T, V} <: AbstractTrustRegion{T, V}
  initial_radius::T
  radius::T
  max_radius::T
  acceptance_threshold::T
  increase_threshold::T
  decrease_factor::T
  increase_factor::T
  ratio::T
  gt::V
  good_grad::Bool

  function TrustRegion(
    gt::V,
    initial_radius::T;
    max_radius::T = one(T) / sqrt(eps(T)),
    acceptance_threshold::T = T(1.0e-4),
    increase_threshold::T = T(0.95),
    decrease_factor::T = one(T) / 3,
    increase_factor::T = 3 * one(T) / 2,
  ) where {T, V}
    initial_radius > 0 || (initial_radius = one(T))
    max_radius > initial_radius || throw(TrustRegionException("Invalid initial radius"))
    (0 < acceptance_threshold < increase_threshold < 1) ||
      throw(TrustRegionException("Invalid thresholds"))
    (0 < decrease_factor < 1 < increase_factor) ||
      throw(TrustRegionException("Invalid decrease/increase factors"))

    return new{T, V}(
      initial_radius,
      initial_radius,
      max_radius,
      acceptance_threshold,
      increase_threshold,
      decrease_factor,
      increase_factor,
      0,
      gt,
      false,
    )
  end
end

TrustRegion(::Type{V}, n::Int, Δ₀::T; kwargs...) where {T, V} = TrustRegion(V(undef, n), Δ₀; kwargs...)
TrustRegion(n::Int, Δ₀::T; kwargs...) where {T} = TrustRegion(Vector{T}, n, Δ₀; kwargs...)

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

function update!(tr::TrustRegion{T, V}, step_norm::T) where {T, V}
  if tr.ratio < tr.acceptance_threshold
    tr.radius = tr.decrease_factor * step_norm
  elseif tr.ratio >= tr.increase_threshold
    tr.radius = min(tr.max_radius, max(tr.radius, tr.increase_factor * step_norm))
  end
  return tr
end
