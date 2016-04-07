# A trust-region solver for unconstrained optimization
# using exact second derivatives.
#
# This implementation follows the description given in [1].
# The main algorithm follows the basic trust-region method described in Section 6.
# The backtracking linesearch follows Section 10.3.2.
# The nonmonotone strategy follows Section 10.1.3, Algorithm 10.1.2.
#
# [1] A. R. Conn, N. I. M. Gould, and Ph. L. Toint,
#     Trust-Region Methods, volume 1 of MPS/SIAM Series on Optimization.
#     SIAM, Philadelphia, USA, 2000.
#     DOI: 10.1137/1.9780898719857.

export trunk

function trunk(nlp :: AbstractNLPModel;
               atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
               max_f :: Int=0,
               bk_max :: Int=5,
               monotone :: Bool=false,
               nm_itmax :: Int=25,
               verbose :: Bool=true)

  n = nlp.meta.nvar
  x = copy(nlp.meta.x0)
  max_f == 0 && (max_f = max(min(100, 2 * n), 5000))
  cgtol = 1.0  # Must be ≤ 1.

  # Armijo linesearch parameter.
  β = 1.0e-4

  iter = 0
  f = obj(nlp, x)
  ∇f = grad(nlp, x)
  ∇fNorm2 = BLAS.nrm2(n, ∇f, 1)
  ∇fNorm = norm(∇f, Inf)
  ϵ = atol + rtol * ∇fNorm
  tr = TrustRegion(min(max(0.1 * ∇fNorm2, 1.0), 100.0))

  # Non-monotone mode parameters.
  # fmin: current best overall objective value
  # nm_iter: number of successful iterations since fmin was first attained
  # fref: objective value at reference iteration
  # σref: cumulative model decrease over successful iterations since the reference iteration
  fmin = fref = fcur = f
  σref = σcur = 0.0
  nm_iter = 0

  # Preallocate xt.
  xt = Array(Float64, n)

  optimal = ∇fNorm2 <= ϵ
  tired = nlp.counters.neval_obj > max_f
  stalled = false

  if verbose
    @printf("%4s  %9s  %7s  %7s  %8s  %5s  %2s  %s\n", "Iter", "f", "‖∇f‖", "Radius", "Ratio", "Inner", "bk", "status")
    @printf("%4d  %9.2e  %7.1e  %7.1e  ", iter, f, ∇fNorm, get_property(tr, :radius))
  end

  while !(optimal || tired || stalled)
    iter = iter + 1

    # Compute inexact solution to trust-region subproblem
    # minimize g's + 1/2 s'Hs  subject to ‖s‖ ≤ radius
    # H = opHermitian(hess(nlp, x))
    H = LinearOperator(n, Float64, v -> hprod(nlp, x, v))
    q = s -> dot(∇f, s) + 0.5 * dot(s, H * s)
    cgtol = max(ϵ, min(0.7 * cgtol, 0.01 * ∇fNorm2))
    (s, cg_stats) = cg(H, -∇f,
                       atol=cgtol, rtol=0.0,
                       radius=get_property(tr, :radius),
                       itmax=max(3 * n, 50),
                       verbose=false)

    # Compute actual vs. predicted reduction.
    sNorm = BLAS.nrm2(n, s, 1)
    BLAS.blascopy!(n, x, 1, xt, 1)
    BLAS.axpy!(n, 1.0, s, 1, xt, 1)
    ft = obj(nlp, xt)
    qs = q(s)

    try
      ρ = ratio(f, ft, qs)
    catch exc # negative predicted reduction
      status = exc.msg
      stalled = true
      continue
    end

    if !monotone
      ρ_hist = ratio(fref, ft, σref + qs)
      ρ = max(ρ, ρ_hist)
    end

    bk = 0
    if acceptable(tr, ρ)
      # Update non-monotone mode parameters.
      if !monotone
        σref = σref + qs
        σcur = σcur + qs
        if ft < fmin
          # New overall best objective value found.
          fcur = ft
          fmin = ft
          σcur = 0.0
          nm_iter = 0
        else
          nm_iter = nm_iter + 1
        end

        if ft > fcur
          fcur = ft
          σcur = 0.0
        end

        if nm_iter >= nm_itmax
          fref = fcur
          σref = σcur
        end
      end
    else
      # Perform backtracking linesearch along s
      # Scaling s to the trust-region boundary, as recommended in
      # Algorithm 10.3.2 of the Trust-Region book
      # appears to deteriorate results.
      # BLAS.scal!(n, get_property(tr, :radius) / sNorm, s, 1)

      slope = BLAS.dot(n, s, 1, ∇f, 1)
      slope < 0.0 || error("not a descent direction: slope = ", slope)
      α = 1.0
      while (bk < bk_max) && (ft > f + β * α * slope)
        bk = bk + 1
        α /= 1.2
        BLAS.blascopy!(n, x, 1, xt, 1)
        BLAS.axpy!(n, α, s, 1, xt, 1)
        ft = obj(nlp, xt)
      end
      sNorm *= α
    end

    verbose && @printf("%8.1e  %5d  %2d  %s\n", ρ, length(cg_stats.residuals), bk, cg_stats.status)

    # Move on.
    update!(tr, ρ, sNorm)
    BLAS.blascopy!(n, xt, 1, x, 1)
    f = ft
    ∇f = grad(nlp, x)
    ∇fNorm2 = BLAS.nrm2(n, ∇f, 1)
    ∇fNorm = norm(∇f, Inf)

    verbose && @printf("%4d  %9.2e  %7.1e  %7.1e  ", iter, f, ∇fNorm2, get_property(tr, :radius))

    optimal = ∇fNorm2 <= ϵ
    tired = nlp.counters.neval_obj > max_f
  end
  verbose && @printf("\n")

  stalled || (status = tired ? "maximum number of evaluations" : "first-order stationary")

  # TODO: create a type to hold solver statistics.
  return (x, f, ∇fNorm, iter, optimal, tired, status)
end
