# A trust-region type and basic utility functions.

include("steihaug.jl")

"Exception type raised in case of error."
type TrustRegionException <: Exception
  msg  :: ASCIIString
end

type TrustRegion
  initial_radius :: Float64
  radius :: Float64
  max_radius :: Float64
  acceptance_threshold :: Float64
  increase_threshold :: Float64
  decrease_factor :: Float64
  increase_factor :: Float64

  function TrustRegion(initial_radius :: Float64;
                       max_radius :: Float64=1.0/sqrt(eps(Float64)),
                       acceptance_threshold :: Float64=1.0e-4,
                       increase_threshold :: Float64=0.95,
                       decrease_factor :: Float64=1./3,
                       increase_factor :: Float64=1.5)

    initial_radius > 0 || (initial_radius = 1.0)
    max_radius > initial_radius || throw(TrustRegionException("Invalid initial radius"))
    (0.0 < acceptance_threshold < increase_threshold < 1.0) || throw(TrustRegionException("Invalid thresholds"))
    (0.0 < decrease_factor < 1.0 < increase_factor) || throw(TrustRegionException("Invalid decrease/increase factors"))

    return new(initial_radius, initial_radius, max_radius,
               acceptance_threshold, increase_threshold,
               decrease_factor, increase_factor)
  end
end

"""Compute the actual vs. predicted reduction radio ∆f/Δm, where
Δf = f_trial - f is the actual reduction is an objective/merit/penalty function,
Δm = m_trial - m is the reduction predicted by the model m of f.
We assume that m is being minimized, and therefore that Δm < 0.
"""
function ratio(f :: Float64, f_trial :: Float64, Δm :: Float64)
  pred = Δm - max(1.0, abs(f)) * 10.0 * eps(Float64)
  pred < 0 || throw(TrustRegionException(@sprintf("Nonnegative predicted reduction: pred = %8.1e", pred)))

  ared = f_trial - f + max(1.0, abs(f)) * 10.0 * eps(Float64)
  return ared / pred
end


"Return `true` if a step is acceptable"
function acceptable(tr :: TrustRegion, ratio :: Float64)
  return ratio >= tr.acceptance_threshold
end


"""Update the trust-region radius based on the ratio of actual vs. predicted
reduction and the step norm. In order to exclude an unsuccessful step, the
update rule is

    radius = decrease_threshold * step_norm                if ratio < acceptance_threshold
    radius = min(max_radius,
                 max(radius, increase_factor * step_norm)  if ratio ≥ increase_threshold
    radius unchanged otherwise.
"""
function update!(tr :: TrustRegion, ratio :: Float64, step_norm :: Float64)
  if ratio < tr.acceptance_threshold
    tr.radius = tr.decrease_factor * step_norm
  elseif ratio >= tr.increase_threshold
    tr.radius = min(tr.max_radius,
                    max(tr.radius, tr.increase_factor * step_norm))
  end
  return tr
end


"Reset the trust-region radius to its initial value"
function reset!(tr :: TrustRegion)
  tr.radius = tr.initial_radius
  return tr
end


"""A basic getter for `TrustRegion` instances.
Should be overhauled when it's possible to overload `getfield()`
and `setfield!()`. See
https://github.com/JuliaLang/julia/issues/1974
"""
function get_property(tr :: TrustRegion, prop :: Symbol)
  # All fields are gettable.
  gettable = fieldnames(TrustRegion)
  prop in gettable || throw(TrustRegionException("Unknown property: $prop"))
  getfield(tr, prop)
end
