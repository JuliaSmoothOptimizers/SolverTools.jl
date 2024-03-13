export armijo_goldstein

"""
    t, ht, nbk, nbG = armijo_goldstein(h, h₀, slope)

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

The method initializes an interval ` [t_low,t_up]` guaranteed to contain a point satifying both Armijo and Goldstein conditions, and then uses a bisection algorithm to find such a point.

# Arguments

- `h::LineModel{T, S, M}`: 1-D model along the search direction `d`, ``h(t) = f(x + t d)``
- `h₀::T`: value of `h` at `t=0`
- `slope`: dot product of the gradient and the search direction, ``∇f(x_k)ᵀd``
The keyword arguments may include
- `t::T = one(T)`: starting steplength (default `1`);
- `τ₀::T = T(eps(T)^(1/4))`: slope factor in the Armijo condition (default `max(1e-4, √ϵₘ)`);
- `τ₁::T = min(prevfloat(T(1)),T(0.9999))`: slope factor in the Goldstein condition. It should satisfy `τ₁ > τ₀` (default `0.9999`);
- `γ₀::T = T(1 / 2)`: backtracking step length mutliplicative factor (0<γ₀<1)
- `γ₁::T = T(2)`: look-aheqd step length mutliplicative factor (1<γ₁)
- `bk_max`: maximum number of backtracks (default `10`);
- `bG_max`: maximum number of increases (default `10`);
- `verbose`: whether to print information (default `false`).

# Outputs

- `t::T`: the step length;
- `ht::T`: the model value at `t`, i.e., `f(x + t * d)`;
- `nbk::Int`: the number of times the steplength was decreased to satisfy the Armijo condition, i.e., number of backtracks;
- `nbG::Int`: the number of times the steplength was increased to satisfy the Goldstein condition.


"""
function armijo_goldstein(
  h::LineModel,
  h₀::T,
  slope::T;
  t::T = one(T),
  τ₀::T = T(eps(T)^(1/4)),
  τ₁::T = min(prevfloat(T(1)),T(0.9999)),
  γ₀::T = T(1 / 2),
  γ₁::T = T(2),
  bk_max::Int = 10,
  bG_max::Int = 10,
  verbose::Bool = false,
) where {T <: AbstractFloat}
  t_low = T(0)
  t_up = t
  nbk = 0
  nbG = 0

  ht = obj(h, t)

  armijo_fail::Bool = !armijo_condition(h₀, ht, τ₀, t, slope)
  goldstein_fail::Bool = !goldstein_condition(h₀, ht, τ₁, t, slope)
  # Backtracking: set t_up so that Armijo condition is satisfied
  if armijo_fail
    while armijo_fail && (nbk < bk_max)
      t_up = t
      t *= γ₀
      ht = obj(h, t)
      nbk += 1
      armijo_fail = !armijo_condition(h₀, ht, τ₀, t, slope)
    end
    goldstein_fail = !goldstein_condition(h₀, ht, τ₁, t, slope)
    t_low = t
    #Look ahead: set t_low so that Goldstein condition is satisfied
  elseif goldstein_fail
    while goldstein_fail && (nbG < bG_max)
      t_low = t
      t *= γ₁
      ht = obj(h, t)
      nbG += 1
      goldstein_fail = !goldstein_condition(h₀, ht, τ₁, t, slope)
    end
    armijo_fail = !armijo_condition(h₀, ht, τ₀, t, slope)
    t_up = t
  else
  end

  # Bisect inside bracket [t_low, t_up]
  
  while (armijo_fail && (nbk < bk_max)) && nbk || (goldstein_fail && (nbG < bG_max))
    t = (t_up - t_low) / 2
    if armijo_fail
      t_up = t
      nbk += 1
    elseif goldstein_fail
      t_low = t
      nbG += 1
    else
    end
    ht = obj(h, t)
    armijo_fail = !armijo_condition(h₀, ht, τ₀, t, slope)
    goldstein_fail = !goldstein_condition(h₀, ht, τ₁, t, slope)
  end

  verbose && @printf("  %4d %4d\n", nbk, nbG)

  return (t, ht, nbk, nbG)
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