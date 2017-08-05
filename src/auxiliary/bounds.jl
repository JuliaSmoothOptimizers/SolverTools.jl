# Auxiliary function related to bound-constrainted problems

export active, breakpoints, compute_Hs_slope_qs, project!, project_step!

"""`active(x, ℓ, u; rtol = 1e-8, atol = 1e-8)`

Computes the active bounds at x, using tolerance `min(rtol * (uᵢ-ℓᵢ), atol)`.
If ℓᵢ or uᵢ is not finite, only `atol` is used.
"""
function active(x::Vector, ℓ::Vector, u::Vector; rtol::Real = sqrt(eps(eltype(x))), atol::Real = sqrt(eps(eltype(x))))
  A = Int[]
  n = length(x)
  for i = 1:n
    δ = -Inf < ℓ[i] < u[i] < Inf ? min(rtol * (u[i] - ℓ[i]), atol) : atol
    if ℓ[i] == x[i] == u[i]
      push!(A, i)
    elseif x[i] <= ℓ[i] + δ
      push!(A, i)
    elseif x[i] >= u[i] - δ
      push!(A, i)
    end
  end
  return A
end

"""
    nbrk, brkmin, brkmax = breakpoints(x, d, ℓ, u)

Find the smallest and largest values of `α` such that `x + αd` lies on the
boundary. `x` is assumed to be feasible. `nbrk` is the number of breakpoints
from `x` in the direction `d`.
"""
function breakpoints(x::Vector, d::Vector, ℓ::Vector, u::Vector)
  pos = find( (d .> 0) .& (x .< u) )
  neg = find( (d .< 0) .& (x .> ℓ) )

  nbrk = length(pos) + length(neg)
  nbrk == 0 && return 0, zero(x), zero(x)

  brkmin = Inf
  brkmax = 0.0
  if length(pos) > 0
    @views steps = (u[pos] - x[pos]) ./ d[pos]
    brkmin = min.(brkmin, minimum(steps))
    brkmax = max.(brkmax, maximum(steps))
  end
  if length(neg) > 0
    @views steps = (ℓ[neg] - x[neg]) ./ d[neg]
    brkmin = min.(brkmin, minimum(steps))
    brkmax = max.(brkmax, maximum(steps))
  end
  return nbrk, brkmin, brkmax
end

"""`slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)`

Computes

    Hs = H * s
    slope = dot(g,s)
    qs = ¹/₂sᵀHs + slope
"""
function compute_Hs_slope_qs!(Hs::Vector, H::Union{AbstractMatrix,AbstractLinearOperator},
                              s::Vector, g::Vector)
  Hs .= H * s
  slope = dot(g,s)
  qs = 0.5 * dot(s, Hs) + slope
  return slope, qs
end

"""`project!(y, x, ℓ, u)`

Projects `x` into bounds `ℓ` and `u`, in the sense of
`yᵢ = max(ℓᵢ, min(xᵢ, uᵢ))`.
"""
function project!(y :: Vector, x :: Vector, ℓ :: Vector, u :: Vector)
  y .= max.(ℓ, min.(x, u))
end

"""`project_step!(y, x, d, ℓ, u)`

Computes the projected direction `y = P(x + d) - x`.
"""
function project_step!(y::Vector, x::Vector, d::Vector, ℓ::Vector, u::Vector)
  y .= x .+ d
  project!(y, y, ℓ, u)
  y .-= x
end

