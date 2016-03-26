export lbfgs

function lbfgs(nlp :: AbstractNLPModel;
               atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
               max_f :: Int=0,
               verbose :: Bool=true,
               mem :: Int=5)

  x = copy(nlp.meta.x0)
  n = nlp.meta.nvar

  xt = Array(Float64, n)
  ∇ft = Array(Float64, n)

  f = obj(nlp, x)
  ∇f = grad(nlp, x)
  H = InverseLBFGSOperator(n, mem, scaling=true)

  ∇fNorm = BLAS.nrm2(n, ∇f, 1)
  ϵ = atol + rtol * ∇fNorm
  max_f == 0 && (max_f = max(min(100, 2 * n), 5000))
  iter = 0

  verbose && @printf("%4s  %8s  %7s  %8s  %4s\n", "iter", "f", "‖∇f‖", "∇f'd", "bk")
  verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)

  optimal = ∇fNorm <= ϵ
  tired = nlp.counters.neval_obj > max_f

  while !(optimal || tired)
    d = - H * ∇f
    slope = BLAS.dot(n, d, 1, ∇f, 1)
    slope < 0.0 || error("Not a descent direction! slope = ", slope)

    verbose && @printf("  %8.1e", slope)

    # Perform improved Armijo linesearch.
    h = C1LineFunction(nlp, x, d)
    t, good_grad, ft, nbk, nbW = armijo_wolfe(h, f, slope, ∇ft, τ₁=0.9999, bk_max=25, verbose=false)

    verbose && @printf("  %4d\n", nbk)

    BLAS.blascopy!(n, x, 1, xt, 1)
    BLAS.axpy!(n, t, d, 1, xt, 1)
    good_grad || (∇ft = grad!(nlp, xt, ∇ft))

    # Update L-BFGS approximation.
    push!(H, t * d, ∇ft - ∇f)

    # Move on.
    x = xt
    f = ft
    BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)
    # norm(∇f) bug: https://github.com/JuliaLang/julia/issues/11788
    ∇fNorm = BLAS.nrm2(n, ∇f, 1)
    iter = iter + 1

    verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)

    optimal = ∇fNorm <= ϵ
    tired = nlp.counters.neval_obj > max_f
  end
  verbose && @printf("\n")

  status = tired ? "maximum number of evaluations" : "first-order stationary"
  return (x, f, ∇fNorm, iter, optimal, tired, status)
end
