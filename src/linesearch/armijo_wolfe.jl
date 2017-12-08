export armijo_wolfe

function armijo_wolfe(h :: LineModel,
                      h₀ :: Float64,
                      slope :: Float64,
                      g :: Array{Float64,1};
                      t :: Float64=1.0,
                      τ₀ :: Float64=1.0e-4,
                      τ₁ :: Float64=0.9999,
                      bk_max :: Int=10,
                      nbWM :: Int=5,
                      verbose :: Bool=false)

  # Perform improved Armijo linesearch.
  nbk = 0
  nbW = 0

  # First try to increase t to satisfy loose Wolfe condition
  ht = obj(h, t)
  slope_t = grad!(h, t, g)
  while (slope_t < τ₁*slope) && (ht <= h₀ + τ₀ * t * slope) && (nbW < nbWM)
    t *= 5.0
    ht = obj(h, t)
    slope_t = grad!(h, t, g)
    nbW += 1
  end

  hgoal = h₀ + slope * t * τ₀;
  fact = -0.8
  ϵ = 1e-10

  # Enrich Armijo's condition with Hager & Zhang numerical trick
  Armijo = (ht <= hgoal) || ((ht <= h₀ + ϵ * abs(h₀)) && (slope_t <= fact * slope))
  good_grad = true
  while !Armijo && (nbk < bk_max)
    t *= 0.4
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
