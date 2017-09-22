using LinearOperators, NLPModels

export ipbox

"""
    ipbox(nlp)

An interior point solver for bound-constrained optimization.
"""
function ipbox(nlp :: AbstractNLPModel; verbose :: Bool=true,
               atol :: Float64=1e-8, rtol :: Float64=1e-6,
               max_f :: Integer=1000,
               max_iter :: Integer=1000,
               max_time :: Float64=60.0,
               τ :: Float64=0.99995 # fraction-to-the-boundary
              )
  start_time = time()
  elapsed_time = 0.0

  l = nlp.meta.lvar
  u = nlp.meta.uvar
  f(x) = obj(nlp, x)
  g(x) = grad(nlp, x)
  n = nlp.meta.nvar
  A = [-speye(n, n)  speye(n, n)]

  P(x, l, u) = min(u, max(l, x))
  x = nlp.meta.x0
  wl = ones(n)
  wu = ones(n)

  μ = 1.0
  σ = 0.5

  fx = f(x)
  gx = g(x)
  B = hess_op(nlp, x)
  πx = norm(P(x - gx, l, u) - x)

  Δ = min(max(1.0, 0.1*πx), 100)
  ϵ = atol + rtol * πx
  iter = 0
  tired = neval_obj(nlp) > max_f || iter > max_iter || elapsed_time > max_time
  optimal = πx < ϵ

  if verbose
    @printf("%4s  %9s  %7s  %7s  %7s\n", "Iter", "f", "π", "Δ", "μ")
    @printf("%-4d  %9.2e  %7.1e  %7.1e  %7.1e\n", iter, fx, πx, Δ, μ)
  end

  while !(optimal || tired)
    ltil = max(x - Δ, l)
    util = min(x + Δ, u)

    D = [-(x-ltil)./wl; -(util-x)./wu]
    H = full(B) - A * diagm(1./D) * A'

    rx = gx + wu - wl
    rl = (x-ltil) .* wl - σ*μ
    ru = (util-x) .* wu - σ*μ

    v = [rl./(x-ltil); ru./(util-x)]
    rhs = A*v - rx
    dx = H\rhs
    dw = -v - (A' * dx)./D
    αx = min( τ * raylength(x, dx, ltil, util), 1.0)
    αw = min( τ * raylength([wl;wu], dw, zeros(2n), Inf*ones(2n)), 1.0)

    xt = x + αx*dx
    ft = f(xt)

    Ared = fx - ft
    Pred = -dot(gx, dx) - 0.5*dot(dx, B * dx)
    ρ = Ared/Pred
    if ρ >= 0.25
      copy!(x, xt)
      fx = ft
      wl += αw * dw[1:n]
      wu += αw * dw[n+1:2n]
      gx = g(x)
      B = hess_op(nlp, x)
      πx = norm(P(x - gx, l, u) - x)
      μ *= 0.1
      if ρ >= 0.75
        Δ *= 4
      end
    else
      Δ /= 4
    end

    iter += 1
    elapsed_time = time() - start_time
    tired = neval_obj(nlp) > max_f || iter > max_iter || elapsed_time > max_time
    optimal = πx < ϵ

    verbose && @printf("%-4d  %9.2e  %7.1e  %7.1e  %7.1e\n", iter, fx, πx, Δ, μ)
  end

  if tired
    status = if neval_obj(nlp) > max_f
      "maximum number of function evaluation"
    elseif iter > max_iter
      "maximum nubmer of iterations"
    elseif elapsed_time > max_time
      "maximum elapsed time"
    end
  else
    status = "first-order stationary"
  end

  return x, f(x), πx, iter, optimal, tired, status, elapsed_time
end
