export armijo_wolfe!

"""
    lso = armijo_wolfe!(ϕ, x, d, xt; kwargs...)

Performs a line search from `x` along the direction `d` for `AbstractMeritModel` `ϕ`.
The steplength is chosen trying to satisfy the Armijo and Wolfe conditions.
The Armijo condition is
```math
ϕ(x + t d; η) ≤ ϕ(x) + τ₀ t Dϕ(x; d, η)
```
and the Wolfe condition is
```math
Dϕ(x + t d; d, η) ≤ τ₁ Dϕ(x; d, η).
```
Initially the step is increased trying to satisfy the Wolfe condition. Afterwards, only backtracking
is performed in order to try to satisfy the Armijo condition. The final steplength may only satisfy
Armijo's condition.

The output is a `LineSearchOutput`. The following specific information are also available in
`lso.specific`:
- nbk: the number of times the steplength was decreased to satisfy the Armijo condition, i.e.,
  number of backtracks;
- nbW: the number of times the steplength was increased to satisfy the Wolfe condition.

The following keyword arguments can be provided:
- `update_obj_at_x`: Whether to call `obj(ϕ, x; update=true)` to update `ϕ.fx` and `ϕ.cx` (default: `false`);
- `update_derivative_at_x`: Whether to call `derivative(ϕ, x, d; update=true)` to update `ϕ.gx` and `ϕ.Ad` (default: `false`);
- `t`: starting steplength (default `1`);
- `τ₀`: slope factor in the Armijo condition (default `max(1e-4, √ϵₘ)`);
- `τ₁`: slope factor in the Wolfe condition. It should satisfy `τ₁ > τ₀` (default `0.9999`);
- `σ`: backtracking decrease factor (default `0.4`);
- `bk_max`: maximum number of backtracks (default `10`);
- `bW_max`: maximum number of increases (default `5`).
"""
function armijo_wolfe!(
  ϕ :: AbstractMeritModel{M,T,V},
  x :: V,
  d :: V,
  xt :: V;
  update_obj_at_x :: Bool = false,
  update_derivative_at_x :: Bool = false,
  t :: T=one(T),
  τ₀ :: T=max(T(1.0e-4), sqrt(eps(T))),
  τ₁ :: T=T(0.9999),
  σ :: T=T(0.4),
  bk_max :: Int=10,
  bW_max :: Int=5
) where {M <: AbstractNLPModel, T <: Real, V <: AbstractVector{<: T}}

  # Perform improved Armijo linesearch.
  nbk = 0
  nbW = 0
  ϕx = obj(ϕ, x; update=update_obj_at_x)
  slope = derivative(ϕ, x, d; update=update_derivative_at_x)

  # First try to increase t to satisfy loose Wolfe condition
  xt .= x .+ t .* d
  ϕt = obj(ϕ, xt)
  slope_t = derivative(ϕ, xt, d)
  while (slope_t < τ₁*slope) && (ϕt <= ϕx + τ₀ * t * slope) && (nbW < bW_max)
    t *= 5
    xt .= x .+ t .* d
    ϕt = obj(ϕ, xt)
    slope_t = derivative(ϕ, xt, d)
    nbW += 1
  end

  ϕgoal = ϕx + slope * t * τ₀;
  fact = -T(0.8)
  ϵ = eps(T)^T(3/5)

  # Enrich Armijo's condition with Hager & Zhang numerical trick
  Armijo = (ϕt <= ϕgoal) || ((ϕt <= ϕx + ϵ * abs(ϕx)) && (slope_t <= fact * slope))
  good_grad = true
  while !Armijo && (nbk < bk_max)
    t *= σ
    xt .= x .+ t .* d
    ϕt = obj(ϕ, xt)
    ϕgoal = ϕx + slope * t * τ₀;

    # avoids unused grad! calls
    Armijo = false
    good_grad = false
    if ϕt <= ϕgoal
      Armijo = true
    elseif ϕt <= ϕx + ϵ * abs(ϕx)
      slope_t = derivative!(ϕ, xt, d)
      good_grad = true
      if slope_t <= fact * slope
        Armijo = true
      end
    end

    nbk += 1
  end

  return LineSearchOutput(t, xt, ϕt, good_grad=good_grad, gt=ϕ.gx, specific=(nbk=nbk, nbW=nbW))
end
