export armijo_goldstein

"""
    t, ht, nbk, nbW = armijo_goldstein(h, h₀, slope)

Performs a line search from `x` along the direction `d` as defined by the
`LineModel` ``h(t) = f(x + t d)``, where
`h₀ = h(0) = f(x)`, `slope = h'(0) = ∇f(x)ᵀd`.
The steplength is chosen trying to satisfy the Armijo and Goldstein conditions. The Armijo condition is
```math
h(t) ≤ h₀ + τ₀ t h'(0)
```
and the Goldstein condition is
```math
h(t) ≥ h₀ + τ₁ t h'(0).
```

The method initializes an interval ` [t_low,t_up]` guaranteed to contain a point satifying both Armijo and Goldstein conditions, and then uses a bissection algorithm to find such a point.

The output is the following:
- t: the step length;
- ht: the model value at `t`, i.e., `f(x + t * d)`;
- nbk: the number of times the steplength was decreased to satisfy the Armijo condition, i.e., number of backtracks;
- nbW: the number of times the steplength was increased to satisfy the Goldstein condition.

The following keyword arguments can be provided:
- `t`: starting steplength (default `1`);
- `τ₀`: slope factor in the Armijo condition (default `max(1e-4, √ϵₘ)`);
- `τ₁`: slope factor in the Goldstein condition. It should satisfy `τ₁ > τ₀` (default `0.9999`);
- `bk_max`: maximum number of backtracks (default `10`);
- `bG_max`: maximum number of increases (default `10`);
- `verbose`: whether to print information (default `false`).
"""
function armijo_goldstein(
  h::LineModel,
  h₀::T,
  slope::T;
  t::T = one(T),
  τ₀::T = T(eps(T)^(1/4)),
  τ₁::T = min(prevfloat(T(1)),T(0.9999)),
  bk_max::Int = 10,
  bW_max::Int = 10,
  verbose::Bool = false,
) where {T <: AbstractFloat}
  t_low = T(0)
  t_up = t
  # Perform improved Armijo linesearch.
  nbk = 0
  nbW = 0

  ht = obj(h, t)

  armijo_fail = !armijo_condition(h₀, ht, τ₀, t, slope)
  goldstein_fail = !goldstein_condition(h₀, ht, τ₁, t, slope)
  # Backtracking: set t_up so that Armijo condition is satisfied
  if armijo_fail
    while armijo_fail && (nbk < bk_max)
      t_up = t
      t /= 2
      ht = obj(h, t)
      nbk += 1
      armijo_fail = !armijo_condition(h₀, ht, τ₀, t, slope)
    end
    t_low = t
    #Look ahead: set t_low so that Goldstein condition is satisfied
  elseif goldstein_fail
    while goldstein_fail && (nbW < bW_max)
      t_low = t
      t *= 2
      ht = obj(h, t)
      nbW += 1
      goldstein_fail = !goldstein_condition(h₀, ht, τ₁, t, slope)
    end
    t_up = t
  else
  end

  # Bisect inside bracket [t_low, t_up]
  armijo_fail = !armijo_condition(h₀, ht, τ₀, t, slope)
  goldstein_fail = !goldstein_condition(h₀, ht, τ₁, t, slope)
  while (armijo_fail && (nbk < bk_max)) && nbk || (goldstein_fail && (nbW < bW_max))
    t = (t_up - t_low) / 2
    if armijo_fail
      t_up = t
      nbk += 1
    elseif goldstein_fail
      t_low = t
      nbW += 1
    else
    end
    ht = obj(h, t)
    armijo_fail = !armijo_condition(h₀, ht, τ₀, t, slope)
    goldstein_fail = !goldstein_condition(h₀, ht, τ₁, t, slope)
  end

  verbose && @printf("  %4d %4d\n", nbk, nbW)

  return (t, ht, nbk, nbW)
end

"""
  armijo_condition(h₀::T, ht::T, τ₀::T, t::T, slope::T)

Returns true if Armijo condition is satisfied for `τ₀`.
""" 
function armijo_condition(h₀::T, ht::T, τ₀::T, t::T, slope::T) where {T}
  ht ≤ h₀ + τ₀ * t * slope
end

"""
  goldstein_condition(h₀::T, ht::T, τ₁::T, t::T, slope::T)

Returns true if Goldstein condition is satisfied for `τ₁`.
""" 
function goldstein_condition(h₀::T, ht::T, τ₁::T, t::T, slope::T) where {T}
  ht ≥ h₀ + τ₁ * t * slope
end