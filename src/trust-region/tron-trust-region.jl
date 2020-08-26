export tron_trust_region!

"""
    tro = tron_trust_region!(ϕ, x, Δx, xt, Δq, Δc, Δ; kwargs...)


Update `xt` and `Δ` using a trust region strategy. See [`trust_region`](@ref) for the basic usage, and below for specifc information about this strategy.

This strategy corresponds to the keyword `method=:tron` on `trust_region`.
In addition to the default keywords, the following keywords are also available:
- `update_derivative_at_x`: Whether to call `derivative(ϕ, x, d; update=true)` to update `ϕ.gx` and `ϕ.Ad` (default: `true`);
- `ηdec`: threshold for decreasing Δ (default: 0.25)
- `σlarge_dec`: factor used in the decrease heuristics (default: 0.25)

Given `ρ = ared / pred`, and the `slope = Dϕ(x, d)`,
- Compute γ = ϕt - ϕx - slope
- Compute α = γ ≤ 0 ? σinc : max(σlarge_dec, -slope / 2γ)
- if `ρ < ηacc`, then `xₖ₊₁ = xₖ` and `Δₖ₊₁ = min(max(α  σlarge_dec)×‖d‖, σdecΔₖ)`;
- if `ηacc ≤ ρ < ηdec`, then `xₖ₊₁ = xₖ + d` and `Δₖ₊₁ = max(σlarge_decΔₖ, min(α‖d‖, σdecΔₖ))`;
- if `ηec ≤ ρ < ηinc`, then `xₖ₊₁ = xₖ + d` and `Δₖ₊₁ = max(σlarge_decΔₖ, min(α‖d‖, σincΔₖ))`;
- otherwise, then `xₖ₊₁ = xₖ + d` and `Δₖ₊₁ = min(Δₘₐₓ, max(Δₖ, min(α‖d‖, σincΔₖ)))`.
"""
function tron_trust_region!(
  ϕ :: AbstractMeritModel{M,T,V},
  x :: V,
  d :: V,
  xt :: V,
  Δq :: T,
  Ad :: V,
  Δ :: T;
  update_obj_at_x :: Bool=false,
  update_derivative_at_x :: Bool=true,
  penalty_update :: Symbol=:basic,
  max_radius :: T=one(T)/sqrt(eps(T)),
  ηacc :: Real=T(1.0e-4),
  ηdec :: Real=T(0.25),
  ηinc :: Real=T(0.95),
  σlarge_dec :: Real=T(0.25),
  σdec :: Real=T(0.5),
  σinc :: Real=3 * one(T) / 2,
) where {M <: AbstractNLPModel, T <: Real, V <: AbstractVector{<: T}}
  (0 < ηacc < ηdec < ηinc < 1) || throw(TrustRegionException("Invalid thresholds"))
  (0 < σlarge_dec < σdec < 1 < σinc) || throw(TrustRegionException("Invalid decrease/increase factors"))
  ϕx = obj(ϕ, x, update=update_obj_at_x)
  slope = derivative(ϕ, x, d, update=update_derivative_at_x)
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
  ϕt = obj(ϕ, xt)
  ϕtf = dualobj(ϕ)
  ϕtc = primalobj(ϕ)

  m = mf + ϕ.η * mc
  ared = ϕx - ϕt
  pred = ϕx - m
  ρ = ared / pred

  γ = ϕt - ϕx - slope
  α = γ ≤ 0 ? σinc : max(σlarge_dec, -slope / 2γ)

  normd = norm(d)
  Δ, status = if ρ < ηacc
    x .= xt
    min(max(α, σlarge_dec) * normd, σdec * Δ), :bad
  elseif ρ < ηdec
    max(σlarge_dec * Δ, min(α * normd, σdec * Δ)), :good
  elseif ρ < ηinc
    max(σlarge_dec * Δ, min(α * normd, σinc * Δ)), :good
  else
    min(max_radius, max(Δ, min(α * normd, σinc * Δ))), :great
  end

  return TrustRegionOutput(status, ared, pred, ρ, status != :bad, Δ, xt, ϕt)
end