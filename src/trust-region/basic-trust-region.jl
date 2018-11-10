export TrustRegion

"""Basic trust region type."""
mutable struct TrustRegion <: AbstractTrustRegion
  initial_radius :: Float64
  radius :: Float64
  max_radius :: Float64
  acceptance_threshold :: Float64
  increase_threshold :: Float64
  decrease_factor :: Float64
  increase_factor :: Float64
  ratio :: Float64

  function TrustRegion(initial_radius :: Float64;
                       max_radius :: Float64=1.0/sqrt(eps(Float64)),
                       acceptance_threshold :: Float64=1.0e-4,
                       increase_threshold :: Float64=0.95,
                       decrease_factor :: Float64=1.0/3,
                       increase_factor :: Float64=1.5)

    initial_radius > 0 || (initial_radius = 1.0)
    max_radius > initial_radius || throw(TrustRegionException("Invalid initial radius"))
    (0.0 < acceptance_threshold < increase_threshold < 1.0) || throw(TrustRegionException("Invalid thresholds"))
    (0.0 < decrease_factor < 1.0 < increase_factor) || throw(TrustRegionException("Invalid decrease/increase factors"))

    return new(initial_radius, initial_radius, max_radius,
               acceptance_threshold, increase_threshold,
               decrease_factor, increase_factor, 0.0)
  end
end

function update!(tr :: TrustRegion, step_norm :: Float64)
  if tr.ratio < tr.acceptance_threshold
    tr.radius = tr.decrease_factor * step_norm
  elseif tr.ratio >= tr.increase_threshold
    tr.radius = min(tr.max_radius,
                    max(tr.radius, tr.increase_factor * step_norm))
  end
  return tr
end
