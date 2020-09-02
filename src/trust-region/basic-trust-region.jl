export basic_trust_region!

"""
    tro = basic_trust_region!(ϕ, x, Δx, xt, Δq, Δc, Δ; kwargs...)

Update `xt` and `Δ` using a trust region strategy. See [`trust_region`](@ref) for the basic usage, and below for specifc information about this strategy.

This strategy corresponds to the keyword `method=:basic` on `trust_region`.

Given `ρ = ared / pred`,
- if `ρ < ηacc`, then `xₖ₊₁ = xₖ` and `Δₖ₊₁ = σdec‖d‖`;
- if `ηacc ≤ ρ < ηinc`, then `xₖ₊₁ = xₖ + d` and `Δₖ₊₁ = Δₖ`;
- otherwise, then `xₖ₊₁ = xₖ + d` and `Δₖ₊₁ = min(Δₘₐₓ, max(Δₖ, σinc‖d‖))`.
"""
function basic_trust_region!(
  ϕ :: AbstractMeritModel{M,T,V},
  x :: V,
  d :: V,
  xt :: V,
  Δq :: T,
  Ad :: V,
  Δ :: T;
  update_obj_at_x :: Bool=false,
  update_derivative_at_x :: Bool=false,
  update_obj_at_xt :: Bool=true,
  ft :: T=T(Inf),
  ct :: V=T[],
  penalty_update :: Symbol=:basic,
  max_radius :: T=one(T)/sqrt(eps(T)),
  ηacc :: Real=T(1.0e-4),
  ηinc :: Real=T(0.95),
  σdec :: Real=one(T) / 3,
  σinc :: Real=3 * one(T) / 2,
) where {M <: AbstractNLPModel, T <: Real, V <: AbstractVector{<: T}}
  (0 < ηacc < ηinc < 1) || throw(TrustRegionException("Invalid thresholds"))
  (0 < σdec < 1 < σinc) || throw(TrustRegionException("Invalid decrease/increase factors"))
  ϕx = obj(ϕ, x, update=update_obj_at_x)
  ϕxf = dualobj(ϕ)
  ϕxc = primalobj(ϕ)
  ϕ.fx += Δq
  ϕ.cx .+= Ad
  mf = dualobj(ϕ)
  mc = primalobj(ϕ)

  if penalty_update == :basic
    while ϕxf - mf < -0.1ϕ.η * (ϕxc - mc) < 0
      ϕ.η *= 2
    end
  else
    throw(TrustRegionException("Unidentified penalty_update $penalty_update"))
  end

  @. xt = x + d
  if !update_obj_at_xt
    ϕ.fx = ft
    ϕ.cx .= ct
  end
  ϕt = obj(ϕ, xt, update=update_obj_at_xt)
  ϕx = ϕxf + ϕ.η * ϕxc
  m = mf + ϕ.η * mc

  absf = abs(ϕx)
  ϵ = eps(T)
  ared = ϕt - ϕx + max(one(T), absf) * 10ϵ
  pred = mf - ϕx - max(one(T), absf) * 10ϵ
  good_grad = if abs(Δq) < 10_000 * ϵ || abs(ared) < 10_000 * ϵ * abs(ϕx)
    slope = derivative(ϕ, x, d, update=update_derivative_at_x)
    ared = (slope + derivative(ϕ, xt, d, update=true)) / 2
    true
  else
    false
  end
  # if pred ≥ 0
  #   status = :neg_pred
  #   stalled = true
  #   continue
  # end
  ρ = ared / pred

  normd = norm(d)
  status = if ρ > ηinc
    Δ = min(max_radius, max(Δ, σinc * normd))
    :great
  elseif ρ < ηacc
    Δ = σdec * normd
    :bad
  else
    :good
  end

  return TrustRegionOutput(status, ared, pred, ρ, status != :bad, Δ, xt, ϕt, gt=ϕ.gx, good_grad=good_grad)
end