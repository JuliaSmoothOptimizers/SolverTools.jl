export ARTrustRegion

"""
ARTrustRegion(α₀::T;kwargs...)

Select the main parameters used in the `TRARC` algorithm with `α₀` as initial TR/ARC parameter.
The keyword arguments are:
- `max_α::T`: Maximum value for `α`. Default `T(1) / sqrt(eps(T))`.
- `acceptance_threshold::T`: Ratio over which the step is successful. Default `T(0.1)`.
- `increase_threshold::T`: Ratio over which we increase `α`. Default `T(0.75)`.
- `reduce_threshold::T`: Ratio under which we decrease `α`. Default `T(0.1)`.
- `increase_factor::T`: Factor of increase of `α`. Default `T(5.0)`.
- `decrease_factor::T`: Factor of decrease of `α`. Default `T(0.1)`.
- `max_unsuccinarow::Int`: Limit on the number of successive unsucessful iterations. Default `30`.

Returns a `ARTrustRegion` structure.

This can be compared to `TrustRegion` or `TRONTrustRegion`.
"""
mutable struct ARTrustRegion{T}
  α₀::T
  α::T
  max_α::T
  acceptance_threshold::T
  increase_threshold::T
  reduce_threshold::T
  increase_factor::T
  decrease_factor::T
  large_decrease_factor::T
  max_unsuccinarow::Int

  function ARTrustRegion(
    α₀::T;
    max_α::T = T(1) / sqrt(eps(T)),
    acceptance_threshold::T = T(0.1),
    increase_threshold::T = T(0.75),
    reduce_threshold::T = T(0.1),
    increase_factor::T = T(5.0),
    decrease_factor::T = T(0.1),
    large_decrease_factor::T = T(0.01),
    max_unsuccinarow::Int = 30,
  ) where {T}
    α₀ > T(0) || (α₀ = T(1))
    max_α > α₀ || throw(TrustRegionException("Invalid α₀"))
    (T(0) < acceptance_threshold < increase_threshold < T(1)) ||
      throw(TrustRegionException("Invalid thresholds"))
    (T(0) < decrease_factor < T(1) < increase_factor) ||
      throw(TrustRegionException("Invalid decrease/increase factors"))

    return new{T}(
      α₀,
      α₀,
      max_α,
      acceptance_threshold,
      increase_threshold,
      reduce_threshold,
      increase_factor,
      decrease_factor,
      large_decrease_factor,
      max_unsuccinarow,
    )
  end
end

"""
    compute_r(nlp, f, Δf, Δq, slope, d, xnext, gnext, robust)

Compute the actual vs predicted reduction ratio `∆f/Δq`.

Arguments:
- `nlp`: Current model we are trying to solve
- `f`: current objective value
- `Δf`: `= f - f_trial` is the actual reduction is an objective/merit/penalty function,
- `Δq`: `q - q_trial` is the reduction predicted by the model q of f.
- `slope`: current slope
- `d`: potential next direction
- `xnext`: potential next iterate
- `gnext`: current gradient value, if `good_grad` is true, then this value has been udpated.
- `robust`: if `true`, try to trap potential cancellation errors

Output:
- `r`: reduction ratio `∆f/Δq`
- `good_grad`: `true` if `gnext` has been recomputed
- `gnext`: gradient.

We assume that `q`` is being minimized, and therefore that `Δq > 0`.
"""
function compute_r(nlp, f::T, Δf, Δq, slope, d, xnext, workspace, robust) where {T}
  gnext = workspace.∇fnext
  good_grad = false
  if robust & ((Δq < 10000 * eps(T)) | (abs(Δf) < 10000 * eps(T) * abs(f)))
    gnext = grad!(nlp, xnext, workspace)
    good_grad = true
    slope_next = dot(gnext, d)

    Δf = -(slope_next + slope) / 2
  end
  r = Δf / Δq
  if isnan(r)
    r = zero(T)
  end

  return r, good_grad, gnext
end
