export lagaug

function lagaug(nlp :: AbstractNLPModel;
               atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
               itmax :: Real=1000, timemax::Real=60.0,
               verbose :: Bool=false)

  if length(nlp.meta.ifree) < nlp.meta.nvar
    error("Not ready")
  elseif nlp.meta.ncon == 0
    return trunk(nlp, atol=atol, rtol=rtol)
  end

  # Let's do no bounds first
  x = nlp.meta.x0
  fx = obj(nlp, x)
  ∇fx = grad(nlp, x)
  cx = cons(nlp, x)
  λ = nlp.meta.y0
  μ = 1.0

  # gpx = ∇f(x) + Aᵀλ
  gpx = ∇fx + jtprod(nlp, x, λ)

  gpxNorm2 = norm(gpx)
  cNorm2 = norm(cx)
  ϵ = atol + rtol * gpxNorm2
  optimal = gpxNorm2 <= ϵ && cNorm2 <= ϵ
  iter = 0
  eltime, start_time = 0.0, time()
  tired = iter >= itmax || eltime >= timemax
  stalled = false

  subtol = 1.0

  if verbose
    @printf("%4s  %9s  %7s  %7s  %7s  %7s  %7s\n", "Iter", "f", "‖gpx‖", "‖cx‖", "μ", "|λ|", "subtol")
    @printf("%4d  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e\n", iter, fx,
            gpxNorm2, cNorm2, μ, norm(λ), subtol)
  end

  iters_since_change = 0
  while !(optimal || tired || stalled)
    iter = iter + 1

    # Compute inexact solution to subproblem
    # minimize f(x) + λᵀc(x) + 0.5μ‖c(x)‖²
    subnlp = subproblem(nlp, x, λ, μ)

    subtol = max(ϵ, min(0.7 * subtol, 0.01 * gpxNorm2))
    x, fx, subπ, iter_sub = trunk(subnlp, rtol=subtol, verbose=false)

    ∇fx = grad(nlp, x)
    cx = cons(nlp, x)
    cNorm2 = norm(cx)

    if cNorm2 < μ^(0.1 + 0.9iters_since_change)
      λ += μ*cx
      iters_since_change += 1
    else
      μ *= 2
      iters_since_change = 0
    end
    gpx = ∇fx + jtprod(nlp, x, λ)

    gpxNorm2 = norm(gpx)
    optimal = gpxNorm2 <= ϵ && cNorm2 <= ϵ
    eltime = time() - start_time
    tired = iter >= itmax || eltime >= timemax
    verbose && @printf("%4d  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e\n", iter,
                       fx, gpxNorm2, cNorm2, μ, norm(λ), subtol)
  end
  verbose && @printf("\n")

  stalled || (status = tired ? "maximum number of evaluations" : "first-order stationary")

  # TODO: create a type to hold solver statistics.
  return (x, fx, gpxNorm2, cNorm2, iter, optimal, tired, status, eltime)
end

"""
subproblem

  minₓ  Lₐ(x) = f(x) + λᵀc(x) + 0.5μ‖c(x)‖^2
"""
function subproblem(nlp::AbstractNLPModel, x::Vector, λ::Vector, μ::Real)
  f(x) = begin
    cx = cons(nlp, x)
    return obj(nlp, x) + dot(λ + 0.5*μ*cx,cx)
  end
  g(x) = grad(nlp, x) + jtprod(nlp, x, λ + μ*cons(nlp, x))
  Hp(x,v;y=[],obj_weight=1.0) = hprod(nlp, x, v, y=λ+μ*cons(nlp, x), obj_weight=obj_weight) +
                                        μ*jtprod(nlp, x, jprod(nlp, x, v))
  return SimpleNLPModel(f, x, g=g, Hp=Hp)
end
