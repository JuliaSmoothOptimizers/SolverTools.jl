# Auxiliary function related to bound-constrainted problems

export active, active!, breakpoints, compute_Hs_slope_qs!, project!, project_step!

"""
    active(x, ℓ, u; rtol = 1e-8, atol = 1e-8)

Computes the active bounds at x, using tolerance `min(rtol * (uᵢ-ℓᵢ), atol)`.
If ℓᵢ or uᵢ is not finite, only `atol` is used.
"""
function active(
  x::AbstractVector{T},
  ℓ::AbstractVector{T},
  u::AbstractVector{T};
  rtol::Real = sqrt(eps(T)),
  atol::Real = sqrt(eps(T)),
) where {T <: Real}
  n = length(x)
  indices = BitVector(undef, n)
  active!(indices, x, ℓ, u, atol = atol, rtol = rtol)
  return findall(indices)
end

"""
    active!(indices, x, ℓ, u; rtol = 1e-8, atol = 1e-8)

Update a `BitVector` of the active bounds at x, using tolerance `min(rtol * (uᵢ-ℓᵢ), atol)`.
If ℓᵢ or uᵢ is not finite, only `atol` is used.
"""
function active!(
  indices::BitVector,
  x::AbstractVector{T},
  ℓ::AbstractVector{T},
  u::AbstractVector{T};
  rtol::Real = sqrt(eps(T)),
  atol::Real = sqrt(eps(T)),
) where {T <: Real}
  n = length(x)
  for i = 1:n
    δ = -Inf < ℓ[i] < u[i] < Inf ? min(rtol * (u[i] - ℓ[i]), atol) : atol
    if ℓ[i] == x[i] == u[i]
      indices[i] = true
    elseif x[i] <= ℓ[i] + δ
      indices[i] = true
    elseif x[i] >= u[i] - δ
      indices[i] = true
    else
      indices[i] = false
    end
  end
  return indices
end

"""
    nbrk, brkmin, brkmax = breakpoints(x, d, ℓ, u)

Find the smallest and largest values of `α` such that `x + αd` lies on the
boundary. `x` is assumed to be feasible. `nbrk` is the number of breakpoints
from `x` in the direction `d`.
"""
function breakpoints(
  x::AbstractVector{T},
  d::AbstractVector{T},
  ℓ::AbstractVector{T},
  u::AbstractVector{T},
) where {T <: Real}
  nvar = length(x)

  nbrk = 0
  brkmin = T(Inf)
  brkmax = zero(T)
  for i=1:nvar
    pos = (d[i] > 0) & (x[i] < u[i])
    if pos
      step = (u[i] - x[i]) / d[i]
      brkmin = min(brkmin, step)
      brkmax = max(brkmax, step)
      nbrk += 1
    end
    neg = (d[i] < 0) & (x[i] > ℓ[i])
    if neg
      step = (ℓ[i] - x[i]) / d[i]
      brkmin = min(brkmin, step)
      brkmax = max(brkmax, step)
      nbrk += 1
    end
  end

  return nbrk, brkmin, brkmax
end

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
    project!(y, x, ℓ, u)

Projects `x` into bounds `ℓ` and `u`, in the sense of
`yᵢ = max(ℓᵢ, min(xᵢ, uᵢ))`.
"""
function project!(
  y::AbstractVector{T},
  x::AbstractVector{T},
  ℓ::AbstractVector{T},
  u::AbstractVector{T},
) where {T <: Real}
  y .= max.(ℓ, min.(x, u))
end

"""
    project_step!(y, x, d, ℓ, u, α = 1.0)

Computes the projected direction `y = P(x + α * d) - x`.
"""
function project_step!(
  y::AbstractVector{T},
  x::AbstractVector{T},
  d::AbstractVector{T},
  ℓ::AbstractVector{T},
  u::AbstractVector{T},
  α::Real = 1.0,
) where {T <: Real}
  y .= x .+ α .* d
  project!(y, y, ℓ, u)
  y .-= x
end
