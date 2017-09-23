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

"Exception type raised in case of error inside Trunk."
type TrunkException <: Exception
  msg  :: String
end

function trunk(nlp :: AbstractNLPModel;
               atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
               max_f :: Int=0,
               max_time :: Float64=Inf,
               bk_max :: Int=10,
               monotone :: Bool=false,
               nm_itmax :: Int=25,
               verbose :: Bool=true)

  start_time = time()
  elapsed_time = 0.0

  x = copy(nlp.meta.x0)
  n = nlp.meta.nvar

  max_f == 0 && (max_f = max(min(100, 2 * n), 5000))
  cgtol = 1.0  # Must be ≤ 1.

  # Armijo linesearch parameter.
  β = 1.0e-4

  iter = 0
  f = obj(nlp, x)
  ∇f = grad(nlp, x)
  ∇fNorm2 = BLAS.nrm2(n, ∇f, 1)
  ϵ = atol + rtol * ∇fNorm2
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
  xt = Array{Float64}(n)
  temp = Array{Float64}(n)

  optimal = ∇fNorm2 <= ϵ
  tired = neval_obj(nlp) > max_f || elapsed_time > max_time
  stalled = false

  if verbose
    @printf("%4s  %9s  %7s  %7s  %8s  %5s  %2s  %s\n", "Iter", "f", "‖∇f‖", "Radius", "Ratio", "Inner", "bk", "status")
    @printf("%4d  %9.2e  %7.1e  %7.1e  ", iter, f, ∇fNorm2, get_property(tr, :radius))
  end

  while !(optimal || tired || stalled)
    iter = iter + 1

    # Compute inexact solution to trust-region subproblem
    # minimize g's + 1/2 s'Hs  subject to ‖s‖ ≤ radius.
    # In this particular case, we may use an operator with preallocation.
    H = hess_op!(nlp, x, temp)
    cgtol = max(ϵ, min(0.7 * cgtol, 0.01 * ∇fNorm2))
    (s, cg_stats) = cg(H, -∇f,
                       atol=cgtol, rtol=0.0,
                       radius=get_property(tr, :radius),
                       itmax=max(2 * n, 50),
                       verbose=false)

    # Compute actual vs. predicted reduction.
    sNorm = BLAS.nrm2(n, s, 1)
    BLAS.blascopy!(n, x, 1, xt, 1)
    BLAS.axpy!(n, 1.0, s, 1, xt, 1)
    slope = BLAS.dot(n, s, 1, ∇f, 1)
    curv = BLAS.dot(n, s, 1, H * s, 1)
    Δq = slope + 0.5 * curv
    ft = obj(nlp, xt)

    try
      ratio!(tr, nlp, f, ft, Δq, xt, s, slope)
    catch exc # negative predicted reduction
      status = :neg_pred
      stalled = true
      continue
    end

    if !monotone
      ρ_hist = ratio(nlp, fref, ft, σref + Δq, xt, s, slope)
      set_property!(tr, :ratio, max(get_property(tr, :ratio), ρ_hist))
    end

    bk = 0
    if !acceptable(tr)
      # Perform backtracking linesearch along s
      # Scaling s to the trust-region boundary, as recommended in
      # Algorithm 10.3.2 of the Trust-Region book
      # appears to deteriorate results.
      # BLAS.scal!(n, get_property(tr, :radius) / sNorm, s, 1)
      # slope *= get_property(tr, :radius) / sNorm
      # sNorm = get_property(tr, :radius)

      slope < 0.0 || throw(TrunkException(@sprintf("not a descent direction: slope = %9.2e, ‖∇f‖ = %7.1e", slope, ∇fNorm2)))
      α = 1.0
      while (bk < bk_max) && (ft > f + β * α * slope)
        bk = bk + 1
        α /= 1.2
        BLAS.blascopy!(n, x, 1, xt, 1)
        BLAS.axpy!(n, α, s, 1, xt, 1)
        ft = obj(nlp, xt)
      end
      sNorm *= α
      BLAS.scal!(n, α, s, 1)
      slope *= α
      Δq = slope + 0.5 * α * α * curv
      ratio!(tr, nlp, f, ft, Δq, xt, s, slope)
      if !monotone
        ρ_hist = ratio(nlp, fref, ft, σref + Δq, xt, s, slope)
        set_property!(tr, :ratio, max(get_property(tr, :ratio), ρ_hist))
      end
    end

    if acceptable(tr)
      # Update non-monotone mode parameters.
      if !monotone
        σref = σref + Δq
        σcur = σcur + Δq
        if ft < fmin
          # New overall best objective value found.
          fcur = ft
          fmin = ft
          σcur = 0.0
          nm_iter = 0
        else
          nm_iter = nm_iter + 1

          if ft > fcur
            fcur = ft
            σcur = 0.0
          end

          if nm_iter >= nm_itmax
            fref = fcur
            σref = σcur
          end
        end
      end

      BLAS.blascopy!(n, xt, 1, x, 1)
      f = ft
      ∇f = grad(nlp, x)
      ∇fNorm2 = BLAS.nrm2(n, ∇f, 1)
    end

    verbose && @printf("%8.1e  %5d  %2d  %s\n", get_property(tr, :ratio), length(cg_stats.residuals), bk, cg_stats.status)

    # Move on.
    update!(tr, sNorm)

    verbose && @printf("%4d  %9.2e  %7.1e  %7.1e  ", iter, f, ∇fNorm2, get_property(tr, :radius))

    optimal = ∇fNorm2 <= ϵ
    elapsed_time = time() - start_time
    tired = neval_obj(nlp) > max_f || elapsed_time > max_time
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

  return ExecutionStats(status, x=x, f=f, normg=∇fNorm2, iter=iter, time=elapsed_time,
                        eval=deepcopy(counters(nlp)))
end
