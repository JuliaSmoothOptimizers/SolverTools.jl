export auglag

include("auglagmodel.jl")

function auglag(nlp :: AbstractNLPModel;
               atol :: Real=1.0e-8, rtol :: Real=1.0e-6,
               max_time :: Real=60.0,
               max_f :: Int=0,
               verbose :: Bool=false,
               subverbose :: Bool=false)

  if nlp.meta.ncon == 0
    return tron(nlp; kwargs...)
  end

  ρ = 1.1
  almodel = AugLagModel(nlp, ρ)

  if max_f == 0; max_f = max(min(100, 2nlp.meta.nvar), 5000); end

  x = nlp.meta.x0
  fx = obj(nlp, x)
  ∇fx = grad(nlp, x)
  cx = cons(nlp, x)
  λ = nlp.meta.y0
  lvar = nlp.meta.lvar
  uvar = nlp.meta.uvar

  # gpx = ∇f(x) + Aᵀλ
  gpx = ∇fx + jtprod(nlp, x, λ)
  project_step!(gpx, x, gpx, lvar, uvar, -1.0)

  gpxNorm2 = norm(gpx)
  cNorm2 = norm(cx)
  ϵ = atol + rtol * gpxNorm2
  optimal = gpxNorm2 <= ϵ && cNorm2 <= ϵ
  iter = 0
  elapsed_time, start_time = 0.0, time()
  tired = neval_obj(nlp) > max_f || elapsed_time >= max_time

  subtol = 1.0/ρ

  if verbose
    @printf("%4s  %9s  %7s  %7s  %7s  %7s  %7s\n", "Iter", "f", "‖gpx‖", "‖cx‖", "ρ", "|λ|", "subtol")
    @printf("%4d  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e\n", iter, fx,
            gpxNorm2, cNorm2, ρ, norm(λ), subtol)
  end

  iters_since_change = 0
  while !(optimal || tired)
    iter = iter + 1

    update_rho(almodel, ρ)
    update_multiplier(almodel, λ)
    subverbose && println("Entering subproblem solver")
    stats = tron(almodel, rtol=subtol, verbose=subverbose)
    subverbose && println("End of subproblem solver")
    x .= stats.solution
    gpxNorm2 = stats.dual_feas

    cx = cons(nlp, x)
    cNorm2 = norm(cx)

    if cNorm2 < 1/ρ^(0.1 + 0.9iters_since_change)
      λ += ρ*cx
      subtol /= ρ
      iters_since_change += 1
    else
      ρ *= 2
      subtol = 1/ρ
      iters_since_change = 0
    end
    fx = obj(nlp, x)

    optimal = gpxNorm2 <= ϵ && cNorm2 <= ϵ
    elapsed_time = time() - start_time
    tired = neval_obj(nlp) > max_f || elapsed_time >= max_time
    verbose && @printf("%4d  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e\n", iter,
                       fx, gpxNorm2, cNorm2, ρ, norm(λ), subtol)
  end
  verbose && @printf("\n")

  if optimal
    status = :first_order
  elseif tired
    if neval_obj(nlp) > max_f
      status = :max_eval
    elseif elapsed_time > max_time
      status = :max_time
    end
  end

  return ExecutionStats(status, solved=optimal, tired=tired, x=x, f=fx,
                        dual_feas=gpxNorm2, primal_feas=cNorm2, iter=iter, time=elapsed_time,
                        eval=deepcopy(counters(nlp)))
end
