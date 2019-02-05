export lbfgs

function lbfgs(nlp :: AbstractNLPModel;
               logger :: AbstractLogger=NullLogger(),
               atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
               max_f :: Int=0,
               max_time :: Float64=Inf,
               verbose :: Bool=true,
               mem :: Int=5)

  start_time = time()
  elapsed_time = 0.0

  x = copy(nlp.meta.x0)
  n = nlp.meta.nvar

  xt = Vector{Float64}(undef, n)
  ∇ft = Vector{Float64}(undef, n)

  f = obj(nlp, x)
  ∇f = grad(nlp, x)
  H = InverseLBFGSOperator(n, mem, scaling=true)

  ∇fNorm = BLAS.nrm2(n, ∇f, 1)
  ϵ = atol + rtol * ∇fNorm
  max_f == 0 && (max_f = max(min(100, 2 * n), 5000))
  iter = 0

  with_logger(logger) do
    @info @sprintf("%4s  %8s  %7s  %8s  %4s", "iter", "f", "‖∇f‖", "∇f'd", "bk")
  end
  infoline = @sprintf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)

  optimal = ∇fNorm ≤ ϵ
  tired = neval_obj(nlp) > max_f || elapsed_time > max_time
  stalled = false
  status = :unknown

  h = LineModel(nlp, x, ∇f)

  while !(optimal || tired || stalled)
    d = - H * ∇f
    slope = BLAS.dot(n, d, 1, ∇f, 1)
    if slope ≥ 0.0
      @error "not a descent direction" slope
      status = :not_desc
      stalled = true
      continue
    end

    infoline *= @sprintf("  %8.1e", slope)

    redirect!(h, x, d)
    # Perform improved Armijo linesearch.
    t, good_grad, ft, nbk, nbW = armijo_wolfe(h, f, slope, ∇ft, τ₁=0.9999, bk_max=25, verbose=false)

    with_logger(logger) do
      @info infoline * @sprintf("  %4d", nbk)
    end

    BLAS.blascopy!(n, x, 1, xt, 1)
    BLAS.axpy!(n, t, d, 1, xt, 1)
    good_grad || grad!(nlp, xt, ∇ft)

    # Update L-BFGS approximation.
    push!(H, t * d, ∇ft - ∇f)

    # Move on.
    x = xt
    f = ft
    BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)
    # norm(∇f) bug: https://github.com/JuliaLang/julia/issues/11788
    ∇fNorm = BLAS.nrm2(n, ∇f, 1)
    iter = iter + 1

    infoline = @sprintf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)

    optimal = ∇fNorm ≤ ϵ
    elapsed_time = time() - start_time
    tired = neval_obj(nlp) > max_f || elapsed_time > max_time
  end
  with_logger(logger) do
    @info infoline
  end

  if optimal
    status = :first_order
  elseif tired
    if neval_obj(nlp) > max_f
      status = :max_eval
    elseif elapsed_time > max_time
      status = :max_time
    end
  end

  return GenericExecutionStats(status, nlp, solution=x, objective=f, dual_feas=∇fNorm,
                               iter=iter, elapsed_time=elapsed_time)
end
