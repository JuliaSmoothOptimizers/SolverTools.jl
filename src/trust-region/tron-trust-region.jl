export TRONTrustRegion

"""Trust region used by TRON"""
mutable struct TRONTrustRegion <: AbstractTrustRegion
  initial_radius :: AbstractFloat
  radius :: AbstractFloat
  max_radius :: AbstractFloat
  acceptance_threshold :: AbstractFloat
  decrease_threshold :: AbstractFloat
  increase_threshold :: AbstractFloat
  large_decrease_factor :: AbstractFloat
  small_decrease_factor :: AbstractFloat
  increase_factor :: AbstractFloat
  ratio :: AbstractFloat
  quad_min :: AbstractFloat

  function TRONTrustRegion(initial_radius :: T;
                           max_radius :: T=one(T)/sqrt(eps(T)),
                           acceptance_threshold :: T=T(1.0e-4),
                           decrease_threshold :: T=T(0.25),
                           increase_threshold :: T=T(0.75),
                           large_decrease_factor :: T=T(0.25),
                           small_decrease_factor :: T=T(0.5),
                           increase_factor :: T=T(4)) where T <: AbstractFloat

    initial_radius > 0 || (initial_radius = one(T))
    max_radius > initial_radius || throw(TrustRegionException("Invalid initial radius"))
    (0 < acceptance_threshold < decrease_threshold < increase_threshold < 1) || throw(TrustRegionException("Invalid thresholds"))
    (0 < large_decrease_factor < small_decrease_factor < 1 < increase_factor) || throw(TrustRegionException("Invalid decrease/increase factors"))

    return new(initial_radius, initial_radius, max_radius,
               acceptance_threshold, decrease_threshold, increase_threshold,
               large_decrease_factor, small_decrease_factor, increase_factor,
               zero(T), zero(T))
  end
end

function aredpred(tr :: TRONTrustRegion, nlp :: AbstractNLPModel, f :: T,
                  f_trial :: T, Δm :: T, x_trial :: Vector{T},
                  step :: Vector{T}, slope :: T) where T <: AbstractFloat
  ared, pred = aredpred(nlp, f, f_trial, Δm, x_trial, step, slope)
  γ = f_trial - f - slope
  quad_min = γ <= 0 ? tr.increase_factor : max(tr.large_decrease_factor, -slope / γ / 2)
  return ared, pred, quad_min
end

function update!(tr :: TRONTrustRegion, step_norm :: AbstractFloat)
  α, σ₁, σ₂, σ₃ = tr.quad_min, tr.large_decrease_factor, tr.small_decrease_factor, tr.increase_factor
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
