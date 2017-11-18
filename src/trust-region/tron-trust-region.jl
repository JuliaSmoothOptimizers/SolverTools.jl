export TRONTrustRegion

"""Trust region used by TRON"""
type TRONTrustRegion <: AbstractTrustRegion
  initial_radius :: Float64
  radius :: Float64
  max_radius :: Float64
  acceptance_threshold :: Float64
  decrease_threshold :: Float64
  increase_threshold :: Float64
  large_decrease_factor :: Float64
  small_decrease_factor :: Float64
  increase_factor :: Float64
  ratio :: Float64
  quad_min :: Float64

  function TRONTrustRegion(initial_radius :: Float64;
                           max_radius :: Float64=1.0/sqrt(eps(Float64)),
                           acceptance_threshold :: Float64=1.0e-4,
                           decrease_threshold :: Float64=0.25,
                           increase_threshold :: Float64=0.75,
                           large_decrease_factor :: Float64=0.25,
                           small_decrease_factor :: Float64=0.5,
                           increase_factor :: Float64=4.0)

    initial_radius > 0 || (initial_radius = 1.0)
    max_radius > initial_radius || throw(TrustRegionException("Invalid initial radius"))
    (0.0 < acceptance_threshold < decrease_threshold < increase_threshold < 1.0) || throw(TrustRegionException("Invalid thresholds"))
    (0.0 < large_decrease_factor < small_decrease_factor < 1.0 < increase_factor) || throw(TrustRegionException("Invalid decrease/increase factors"))

    return new(initial_radius, initial_radius, max_radius,
               acceptance_threshold, decrease_threshold, increase_threshold,
               large_decrease_factor, small_decrease_factor, increase_factor,
               0.0, 0.0)
  end
end

function aredpred(tr :: TRONTrustRegion, nlp :: AbstractNLPModel, f :: Float64,
                  f_trial :: Float64, Δm :: Float64, x_trial :: Vector{Float64},
                  step :: Vector{Float64}, slope :: Float64)
  ared, pred = aredpred(nlp, f, f_trial, Δm, x_trial, step, slope)
  γ = f_trial - f - slope
  quad_min = γ <= 0.0 ? tr.increase_factor : max(tr.large_decrease_factor, -0.5 * slope / γ)
  return ared, pred, quad_min
end

function update!(tr :: TRONTrustRegion, step_norm :: Float64)
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
