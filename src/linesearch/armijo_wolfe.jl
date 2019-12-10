export armijo_wolfe

"""
    t, good_grad, ht, nbk, nbW = armijo_wolfe(h, h₀, slope, g)

Performs a line search from `x` along the direction `d` as defined by the
`LineModel` ``h(t) = f(x + t d)``, where
`h₀ = h(0) = f(x)`, `slope = h'(0) = ∇f(x)ᵀd` and `g` is a vector that will be
overwritten with the gradient at various points. On exit, if `good_grad=true`,
`g` contains the gradient at the final step length.
The steplength is chosen trying to satisfy the Armijo and Wolfe conditions. The Armijo condition is
```math
h(t) ≤ h₀ + τ₀ t h'(0)
```
and the Wolfe condition is
```math
h'(t) ≤ τ₁ h'(0).
```
Initially the step is increased trying to satisfy the Wolfe condition.
Afterwards, only backtracking is performed in order to try to satisfy the Armijo condition.
The final steplength may only satisfy Armijo's condition.

The output is the following:
- t: the step length;
- good_grad: whether `g` is the gradient at `x + t * d`;
- ht: the model value at `t`, i.e., `f(x + t * d)`;
- nbk: the number of times the steplength was decreased to satisfy the Armijo condition, i.e., number of backtracks;
- nbW: the number of times the steplength was increased to satisfy the Wolfe condition.

The following keyword arguments can be provided:
- `t`: starting steplength (default `1`);
- `τ₀`: slope factor in the Armijo condition (default `max(1e-4, √ϵₘ)`);
- `τ₁`: slope factor in the Wolfe condition. It should satisfy `τ₁ > τ₀` (default `0.9999`);
- `bk_max`: maximum number of backtracks (default `10`);
- `bW_max`: maximum number of increases (default `5`);
- `verbose`: whether to print information (default `false`).
"""
function armijo_wolfe(h :: LineModel,
                      h₀ :: T,
                      slope :: T,
                      g :: Array{T,1};
                      t :: T=one(T),
                      τ₀ :: T=max(T(1.0e-4), sqrt(eps(T))),
                      τ₁ :: T=T(0.9999),
                      bk_max :: Int=10,
                      bW_max :: Int=5,
                      verbose :: Bool=false) where T <: AbstractFloat

  # Perform improved Armijo linesearch.
  nbk = 0
  nbW = 0

  # First try to increase t to satisfy loose Wolfe condition
  ht = obj(h, t)
  slope_t = grad!(h, t, g)
  while (slope_t < τ₁*slope) && (ht <= h₀ + τ₀ * t * slope) && (nbW < bW_max)
    t *= 5
    ht = obj(h, t)
    slope_t = grad!(h, t, g)
    nbW += 1
  end

  hgoal = h₀ + slope * t * τ₀;
  fact = -T(0.8)
  ϵ = eps(T)^T(3/5)

  # Enrich Armijo's condition with Hager & Zhang numerical trick
  Armijo = (ht <= hgoal) || ((ht <= h₀ + ϵ * abs(h₀)) && (slope_t <= fact * slope))
  good_grad = true
  while !Armijo && (nbk < bk_max)
    t *= T(0.4)
    ht = obj(h, t)
    hgoal = h₀ + slope * t * τ₀;

    # avoids unused grad! calls
    Armijo = false
    good_grad = false
    if ht <= hgoal
      Armijo = true
    elseif ht <= h₀ + ϵ * abs(h₀)
      slope_t = grad!(h, t, g)
      good_grad = true
      if slope_t <= fact * slope
        Armijo = true
      end
    end

    nbk += 1
  end

  verbose && @printf("  %4d %4d\n", nbk, nbW);

  return (t, good_grad, ht, nbk, nbW)
end
