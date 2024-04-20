export armijo_goldstein

"""
    t, ht, nbk, nbG = armijo_goldstein(h, h₀, slope)

Perform a line search from `x` along the direction `d` as defined by the `LineModel` `h(t) = f(x + t d)`, where `h₀ = h(0) = f(x)`, and `slope = h'(0) = ∇f(x)ᵀd`.
The steplength is chosen to satisfy the Armijo-Goldstein conditions.
The Armijo condition is
```math
h(t) ≤ h₀ + τ₀ t h'(0)
```
and the Goldstein condition is
```math
h(t) ≥ h₀ + τ₁ t h'(0).
```
with `0 < τ₀ < τ₁ < 1`.

# Arguments

- `h::LineModel{T, S, M}`: 1-D model along the search direction `d`, ``h(t) = f(x + t d)``
- `h₀::T`: value of `h` at `t = 0`
- `slope`: dot product of the gradient and the search direction, ``∇f(x)ᵀd``

# Keyword arguments

- `t::T = one(T)`: initial steplength (default: `1`);
- `τ₀::T = T(eps(T)^(1/4))`: slope factor in the Armijo condition. It should satisfy `0 < τ₀ < τ₁ < 1 `;
- `τ₁::T = min(T(1)-eps(T), T(0.9999))`: slope factor in the Goldstein condition. It should satisfy `0 < τ₀ < τ₁ < 1 `;
- `γ₀::T = T(1 / 2)`: backtracking step length mutliplicative factor (0 < γ₀ <1)
- `γ₁::T = T(2)`: look-ahead step length mutliplicative factor (γ₁ > 1)
- `bk_max`: maximum number of backtracks (default: `10`);
- `bG_max`: maximum number of increases (default: `10`);
- `verbose`: whether to print information (default: `false`).

# Outputs

- `t::T`: the step length;
- `ht::T`: the model value at `t`, i.e., `f(x + t * d)`;
- `nbk::Int`: the number of times the steplength was decreased to satisfy the Armijo condition, i.e., the number of backtracks;
- `nbG::Int`: the number of times the steplength was increased to satisfy the Goldstein condition.


# References

This implementation follows the description given in

    C. Cartis, P. R. Sampaio, Ph. L. Toint,
    Worst-case evaluation complexity of non-monotone gradient-related algorithms for unconstrained optimization.
    Optimization 64(5), 1349–1361 (2015).
    DOI: 10.1080/02331934.2013.869809
    
  The method initializes an interval ` [t_low,t_up]` guaranteed to contain a point satifying both Armijo and Goldstein conditions, and then uses a bisection algorithm to find such a point.
  The method is implemented with M=0 (see reference), i.e., Armijo and Goldstein conditions are satisfied only for the current value of the objective `h₀`. 

  # Examples

```jldoctest; output = false
using SolverTools, ADNLPModels
nlp = ADNLPModel(x -> x[1]^2 + 4 * x[2]^2, ones(2))
lm = LineModel(nlp, nlp.meta.x0, -ones(2))

t, ft, nbk, nbG = armijo_goldstein(lm, obj(lm, 0.0), grad(lm, 0.0))

# output

(1.0, 0.0, 0, 0)
```

"""
function armijo_goldstein(
  h::LineModel,
  h₀::T,
  slope::T;
  t::T = one(T),
  τ₀::T = T(eps(T)^(1 / 4)),
  τ₁::T = T(1) - eps(T),
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
  end

  # Bisect inside bracket [t_low, t_up]
  while (armijo_fail && (nbk < bk_max)) || (goldstein_fail && (nbG < bG_max))
    t = (t_up - t_low) / 2
    if armijo_fail
      t_up = t
      nbk += 1
    elseif goldstein_fail
      t_low = t
      nbG += 1
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
