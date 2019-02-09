export TrustRegion

"""Basic trust region type."""
mutable struct TrustRegion <: AbstractTrustRegion
  initial_radius :: AbstractFloat
  radius :: AbstractFloat
  max_radius :: AbstractFloat
  acceptance_threshold :: AbstractFloat
  increase_threshold :: AbstractFloat
  decrease_factor :: AbstractFloat
  increase_factor :: AbstractFloat
  ratio :: AbstractFloat

  function TrustRegion(initial_radius :: T;
                       max_radius :: T=one(T)/sqrt(eps(T)),
                       acceptance_threshold :: T=T(1.0e-4),
                       increase_threshold :: T=T(0.95),
                       decrease_factor :: T=one(T)/3,
                       increase_factor :: T=3 * one(T) / 2) where T <: Number

    initial_radius > 0 || (initial_radius = one(T))
    max_radius > initial_radius || throw(TrustRegionException("Invalid initial radius"))
    (0 < acceptance_threshold < increase_threshold < 1) || throw(TrustRegionException("Invalid thresholds"))
    (0 < decrease_factor < 1 < increase_factor) || throw(TrustRegionException("Invalid decrease/increase factors"))

    return new(initial_radius, initial_radius, max_radius,
               acceptance_threshold, increase_threshold,
               decrease_factor, increase_factor, 0)
  end
end

function update!(tr :: TrustRegion, step_norm :: AbstractFloat)
  if tr.ratio < tr.acceptance_threshold
    tr.radius = tr.decrease_factor * step_norm
  elseif tr.ratio >= tr.increase_threshold
    tr.radius = min(tr.max_radius,
                    max(tr.radius, tr.increase_factor * step_norm))
  end
  return tr
end
