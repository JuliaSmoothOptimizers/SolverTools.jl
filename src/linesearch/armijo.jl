export armijo!

"""
    lso = armijo!(ϕ, x, d, xt; kwargs...)

Performs a line search from `x` along the direction `d` for `AbstractMeritModel` `ϕ`.
The steplength is chosen trying to satisfy the Armijo condition
```math
ϕ(x + t d; η) ≤ ϕ(x) + τ₀ t Dϕ(x; η),
```
using backtracking.

The keyword aguments are:
- ...

The output is a `LineSearchOutput`. The following specific information are also available in
`lso.specific`:
- nbk: the number of times the steplength was decreased to satisfy the Armijo condition, i.e.,
  number of backtracks;

The following keyword arguments can be provided:
- `update_obj_at_x`: Whether to call `obj(ϕ, x; update=true)` to update `ϕ.fx` and `ϕ.cx` (default: `false`);
- `update_derivative_at_x`: Whether to call `derivative(ϕ, x, d; update=true)` to update `ϕ.gx` and `ϕ.Ad` (default: `false`);
- `t`: starting steplength (default `1`);
- `τ₀`: slope factor in the Armijo condition (default `max(1e-4, √ϵₘ)`);
- `σ`: backtracking decrease factor (default `0.4`);
- `bk_max`: maximum number of backtracks (default `10`).
"""
function armijo!(
  ϕ :: AbstractMeritModel{M,T,V},
  x :: V,
  d :: V,
  xt :: V;
  update_obj_at_x :: Bool = false,
  update_derivative_at_x :: Bool = false,
  t :: T=one(T),
  τ₀ :: T=max(T(1.0e-4), sqrt(eps(T))),
  σ :: T=T(0.4),
  bk_max :: Int=10
) where {M <: AbstractNLPModel, T <: Real, V <: AbstractVector{<: T}}

  nbk = 0
  ϕx = obj(ϕ, x; update=update_obj_at_x)
  slope = derivative(ϕ, x, d; update=update_derivative_at_x)

  xt .= x .+ t .* d
  ϕt = obj(ϕ, xt)
  while !(ϕt ≤ ϕx + τ₀ * t * slope) && (nbk < bk_max)
    t *= σ
    xt .= x .+ t .* d
    ϕt = obj(ϕ, xt)

    nbk += 1
  end

  return LineSearchOutput(t, xt, ϕt, specific=(nbk=nbk,))
end
