# Auxiliary function related to bound-constrainted problems

export active,
  active!, breakpoints, compute_Hs_slope_qs!, compute_As_slope_qs!, project!, project_step!

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
  indices = similar(x, Bool)
  active!(indices, x, ℓ, u, atol = atol, rtol = rtol)
  return findall(indices)
end

"""
    active!(indices, x, ℓ, u; rtol = 1e-8, atol = 1e-8)

Update a vector of `Bool` (e.g., a `BitVector`) of the active bounds at x, using tolerance
`min(rtol * (uᵢ-ℓᵢ), atol)`.
If ℓᵢ or uᵢ is not finite, only `atol` is used.
"""
function active!(
  indices::AbstractVector{Bool},
  x::AbstractVector{T},
  ℓ::AbstractVector{T},
  u::AbstractVector{T};
  rtol::Real = sqrt(eps(T)),
  atol::Real = sqrt(eps(T)),
) where {T <: Real}
  indices .= _isactive.(x, ℓ, u, rtol, atol)
  return indices
end

function _isactive(xi, ℓi, ui, rtol, atol)
  δ = -Inf < ℓi < ui < Inf ? min(rtol * (ui - ℓi), atol) : atol
  return (ℓi == xi == ui) || (xi <= ℓi + δ) || (xi >= ui - δ)
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
  # Written as a single reduction over a lazy broadcast so that it does not
  # allocate and does not scalar-index GPU arrays.
  bc = Broadcast.instantiate(Broadcast.broadcasted(_breakpoint, x, d, ℓ, u))
  nbrk, brkmin, brkmax = mapreduce(identity, _merge_breakpoints, bc, init = (0, T(Inf), zero(T)))

  return nbrk, brkmin, brkmax
end

# Breakpoint contribution of one coordinate as `(nbrk, brkmin, brkmax)`.
function _breakpoint(xi::T, di::T, ℓi::T, ui::T) where {T}
  pos = (di > 0) & (xi < ui)
  neg = (di < 0) & (xi > ℓi)
  step = pos ? (ui - xi) / di : (ℓi - xi) / di
  if pos | neg
    return (1, step, step)
  else
    return (0, T(Inf), zero(T))
  end
end

function _merge_breakpoints(a, b)
  return (a[1] + b[1], min(a[2], b[2]), max(a[3], b[3]))
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
