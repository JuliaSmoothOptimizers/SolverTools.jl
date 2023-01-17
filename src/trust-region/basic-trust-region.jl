export TrustRegion

"""
    TrustRegion{T, V} <: AbstractTrustRegion{T, V}

Basic trust region type that contains the following fields:
- `initial_radius::T`: initial radius;
- `radius::T`: current radius;
- `max_radius::T`: upper bound on the radius (default `1 / sqrt(eps(T))`);
- `acceptance_threshold::T`: decrease radius if ratio is below this threshold between 0 and 1 (default `1e-4`);
- `increase_threshold::T`: increase radius if ratio is beyond this threshold between 0 and 1  (default `0.95`);
- `decrease_factor::T`: decrease factor less between 0 and 1 (default `1 / 3`);
- `increase_factor::T`: increase factor greater than one (default `3 / 2`);
- `ratio::T`: current ratio `ared / pred`;
- `gt::V`: pre-allocated memory vector to store the gradient of the objective function;
- `good_grad::Bool`: `true` if `gt` is the gradient of the objective function at the trial point.

The following constructors are available:

    TrustRegion(gt,::V initial_radius::T; kwargs...)

If `gt` is not known, it is possible to use the following constructors:

    TrustRegion(::Type{V}, n::Int, Δ₀::T; kwargs...)
    TrustRegion(n::Int, Δ₀::T; kwargs...)

that will allocate a vector of size `n` and type `V` or `Vector{T}`.
"""
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
    max_radius ≥ initial_radius || throw(TrustRegionException("Invalid initial radius"))
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

TrustRegion(::Type{V}, n::Int, Δ₀::T; kwargs...) where {T, V} =
  TrustRegion(V(undef, n), Δ₀; kwargs...)
TrustRegion(n::Int, Δ₀::T; kwargs...) where {T} = TrustRegion(Vector{T}, n, Δ₀; kwargs...)

function update!(tr::TrustRegion{T, V}, step_norm::T) where {T, V}
  if tr.ratio < tr.acceptance_threshold
    tr.radius = tr.decrease_factor * step_norm
  elseif tr.ratio >= tr.increase_threshold
    tr.radius = min(tr.max_radius, max(tr.radius, tr.increase_factor * step_norm))
  end
  return tr
end
