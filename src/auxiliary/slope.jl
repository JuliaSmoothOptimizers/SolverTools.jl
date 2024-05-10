"""
    slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)

Computes

    Hs = H * s
    slope = dot(g,s)
    qs = ¹/₂sᵀHs + slope
"""
function compute_Hs_slope_qs!(
  Hs::AbstractVector{T},
  H::Union{AbstractMatrix, AbstractLinearOperator},
  s::AbstractVector{T},
  g::AbstractVector{T},
) where {T <: Real}
  mul!(Hs, H, s)
  slope = dot(g, s)
  qs = dot(s, Hs) / 2 + slope
  return slope, qs
end

"""
    slope, qs = compute_As_slope_qs!(As, A, s, Fx)

Compute `slope = dot(As, Fx)` and `qs = dot(As, As) / 2 + slope`. Use `As` to store `A * s`.
"""
function compute_As_slope_qs!(
  As::AbstractVector{T},
  A::Union{AbstractMatrix, AbstractLinearOperator},
  s::AbstractVector{T},
  Fx::AbstractVector{T},
) where {T <: Real}
  mul!(As, A, s)
  slope = dot(As, Fx)
  qs = dot(As, As) / 2 + slope
  return slope, qs
end
