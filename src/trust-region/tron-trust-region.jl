export TRONTrustRegion

"""Trust region used by TRON"""
mutable struct TRONTrustRegion{T, V} <: AbstractTrustRegion{T, V}
  initial_radius::T
  radius::T
  max_radius::T
  acceptance_threshold::T
  decrease_threshold::T
  increase_threshold::T
  large_decrease_factor::T
  small_decrease_factor::T
  increase_factor::T
  ratio::T
  quad_min::T
  gt::V
  good_grad::Bool

  function TRONTrustRegion(
    gt::V,
    initial_radius::T;
    max_radius::T = one(T) / sqrt(eps(T)),
    acceptance_threshold::T = T(1.0e-4),
    decrease_threshold::T = T(0.25),
    increase_threshold::T = T(0.75),
    large_decrease_factor::T = T(0.25),
    small_decrease_factor::T = T(0.5),
    increase_factor::T = T(4),
  ) where {T, V}
    initial_radius > 0 || (initial_radius = one(T))
    max_radius > initial_radius || throw(TrustRegionException("Invalid initial radius"))
    (0 < acceptance_threshold < decrease_threshold < increase_threshold < 1) ||
      throw(TrustRegionException("Invalid thresholds"))
    (0 < large_decrease_factor < small_decrease_factor < 1 < increase_factor) ||
      throw(TrustRegionException("Invalid decrease/increase factors"))

    return new{T, V}(
      initial_radius,
      initial_radius,
      max_radius,
      acceptance_threshold,
      decrease_threshold,
      increase_threshold,
      large_decrease_factor,
      small_decrease_factor,
      increase_factor,
      zero(T),
      zero(T),
      gt,
      false,
    )
  end
end

TRONTrustRegion(::Type{V}, n::Int, Δ₀::T; kwargs...) where {T, V} = TRONTrustRegion(V(undef, n), Δ₀; kwargs...)
TRONTrustRegion(n::Int, Δ₀::T; kwargs...) where {T} = TRONTrustRegion(Vector{T}, n, Δ₀; kwargs...)

function aredpred!(
  tr::TRONTrustRegion{T, V},
  nlp::AbstractNLPModel{T, V},
  f::T,
  f_trial::T,
  Δm::T,
  x_trial::V,
  step::V,
  slope::T,
) where {T, V}
  ared, pred, tr.good_grad = aredpred_common(nlp, f, f_trial, Δm, x_trial, step, tr.gt, slope)
  γ = f_trial - f - slope
  tr.quad_min = γ <= 0 ? tr.increase_factor : max(tr.large_decrease_factor, -slope / γ / 2)
  return ared, pred
end

function update!(tr::TRONTrustRegion{T, V}, step_norm::T) where {T, V}
  α, σ₁, σ₂, σ₃ =
    tr.quad_min, tr.large_decrease_factor, tr.small_decrease_factor, tr.increase_factor
  tr.radius = if tr.ratio < tr.acceptance_threshold
    min(max(α, σ₁) * step_norm, σ₂ * tr.radius)
  elseif tr.ratio < tr.decrease_threshold
    max(σ₁ * tr.radius, min(α * step_norm, σ₂ * tr.radius))
  elseif tr.ratio < tr.increase_threshold
    max(σ₁ * tr.radius, min(α * step_norm, σ₃ * tr.radius))
  else
    min(tr.max_radius, max(tr.radius, min(α * step_norm, σ₃ * tr.radius)))
  end
  return tr
end
