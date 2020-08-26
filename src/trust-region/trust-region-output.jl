export TrustRegionOutput

struct TrustRegionOutput{T <: Real, V <: AbstractVector}
  status :: Symbol
  ared :: T
  pred :: T
  ρ :: T
  success :: Bool
  Δ :: T
  xt :: V
  ϕt :: T
  specific :: NamedTuple
end

function TrustRegionOutput(
  status :: Symbol,
  ared :: T,
  pred :: T,
  ρ :: T,
  success :: Bool,
  Δ :: T,
  xt :: V,
  ϕt :: T;
  specific :: NamedTuple = NamedTuple()
) where {T <: Real, V <: AbstractVector{<: T}}
  TrustRegionOutput(status, ared, pred, ρ, success, Δ, xt, ϕt, specific)
end