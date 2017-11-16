# A trust-region solver for nonlinear least squares.
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

const trunkls_allowed_subsolvers = [:cgls, :crls, :lsqr, :lsmr]

function trunk(nlp :: AbstractNLSModel;
               subsolver :: Symbol=:lsmr,
               atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
               max_f :: Int=0,
               max_time :: Float64=Inf,
               bk_max :: Int=10,
               monotone :: Bool=true,
               nm_itmax :: Int=25,
               trsolver_args :: Dict{Symbol,Any}=Dict{Symbol,Any}())

  start_time = time()
  elapsed_time = 0.0

  subsolver in trunkls_allowed_subsolvers || error("subproblem solver must be one of $(trunkls_allowed_subsolvers)")
  trsolver = eval(subsolver)
  n = nlp.nls_meta.nvar
  m = nlp.nls_meta.nequ
  x = copy(nlp.meta.x0)
  max_f == 0 && (max_f = max(min(100, 2 * n), 5000))
  cgtol = 1.0  # Must be ≤ 1.

  # Armijo linesearch parameter.
  β = 1.0e-4

  iter = 0
  r = residual(nlp, x)
  f = 0.5 * dot(r, r)

  # preallocate storage for products with A and A'
  Av = Vector{Float64}(undef, m)
  Atv = Vector{Float64}(undef, n)
  A = jac_op_residual!(nlp, x, Av, Atv)
  ∇f = A' * r
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
  xt = Vector{Float64}(undef, n)
  temp = Vector{Float64}(undef, n)

  optimal = ∇fNorm2 ≤ ϵ
  tired = nlp.counters.neval_residual > max_f || elapsed_time > max_time
  stalled = false
  status = :unknown

  @info @sprintf("%4s  %9s  %7s  %7s  %7s  %8s  %5s  %2s  %s",
                 "Iter", "f", "‖∇f‖", "Radius", "Step", "Ratio", "Inner", "bk", "status")
  infoline = @sprintf("%4d  %9.2e  %7.1e  %7.1e  ",
                      iter, f, ∇fNorm2, get_property(tr, :radius))

  while !(optimal || tired || stalled)
    iter = iter + 1

    # Compute inexact solution to trust-region subproblem
    # minimize g's + 1/2 s'Hs  subject to ‖s‖ ≤ radius.
    # In this particular case, we may use an operator with preallocation.
    cgtol = max(rtol, min(0.1, 0.9 * cgtol, sqrt(∇fNorm2)))
    (s, cg_stats) = trsolver(A, -r,
                             atol=atol, rtol=cgtol,
                             radius=get_property(tr, :radius),
                             itmax=max(2 * (n + m), 50), verbose=false;
                             trsolver_args...)

    # Compute actual vs. predicted reduction.
    sNorm = BLAS.nrm2(n, s, 1)
    BLAS.blascopy!(n, x, 1, xt, 1)
    BLAS.axpy!(n, 1.0, s, 1, xt, 1)
    # slope = dot(∇f, s)
    t = A * s
    slope = dot(r, t)
    curv = dot(t, t)
    Δq = slope + 0.5 * curv
    rt = residual(nlp, xt)
    ft = 0.5 * dot(rt, rt)
    @debug @sprintf("‖s‖ = %7.1e, slope = %8.1e, Δq = %15.7e", sNorm, slope, Δq)

    ared, pred = aredpred(nlp, f, ft, Δq, xt, s, slope)
    if pred ≥ 0
      infoline *= @sprintf("%7.1e  %8.1e  %5d  %2d  %s",
                           sNorm, get_property(tr, :ratio), length(cg_stats.residuals),
                           0, cg_stats.status)
      status = :neg_pred
      stalled = true
      continue
    end
    tr.ratio = ared / pred

    if !monotone
      ared_hist, pred_hist = aredpred(nlp, fref, ft, σref + Δq, xt, s, slope)
      if pred_hist ≥ 0
        infoline *= @sprintf("%7.1e  %8.1e  %5d  %2d  %s",
                             sNorm, get_property(tr, :ratio), length(cg_stats.residuals),
                             0, cg_stats.status)
        status = :neg_pred
        stalled = true
        continue
      end
      ρ_hist = ared_hist / pred_hist
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

      if slope ≥ 0
        @error "not a descent direction" slope ∇fNorm2 sNorm
        status = :not_desc
        stalled = true
        continue
      end
      α = 1.0
      while (bk < bk_max) && (ft > f + β * α * slope)
        bk = bk + 1
        α /= 1.2
        BLAS.blascopy!(n, x, 1, xt, 1)
        BLAS.axpy!(n, α, s, 1, xt, 1)
        rt = residual(nlp, xt)
        ft = 0.5 * dot(rt, rt)
        @debug "" α ft
      end
      sNorm *= α
      BLAS.scal!(n, α, s, 1)
      slope *= α
      Δq = slope + 0.5 * α * α * curv
      @debug "" slope Δq
      ared, pred = aredpred(nlp, f, ft, Δq, xt, s, slope)
      if pred ≥ 0
        infoline *= @sprintf("%7.1e  %8.1e  %5d  %2d  %s",
                             sNorm, get_property(tr, :ratio), length(cg_stats.residuals),
                             bk, cg_stats.status)
        status = :neg_pred
        stalled = true
        continue
      end
      tr.ratio = ared / pred
      if !monotone
        ared_hist, pred_hist = ratio(nlp, fref, ft, σref + Δq, xt, s, slope)
        if pred_hist ≥ 0
          infoline *= @sprintf("%7.1e  %8.1e  %5d  %2d  %s",
                               sNorm, get_property(tr, :ratio), length(cg_stats.residuals),
                               bk, cg_stats.status)
          status = :neg_pred
          stalled = true
          continue
        end
        ρ_hist = ared_hist / pred_hist
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
      r = rt
      f = ft
      A = jac_op_residual!(nlp, x, Av, Atv)
      ∇f = A' * r
      ∇fNorm2 = BLAS.nrm2(n, ∇f, 1)
    end

    infoline *= @sprintf("%7.1e  %8.1e  %5d  %2d  %s",
                         sNorm, get_property(tr, :ratio), length(cg_stats.residuals),
                         bk, cg_stats.status)
    @info infoline

    # Move on.
    update!(tr, sNorm)

    infoline = @sprintf("%4d  %9.2e  %7.1e  %7.1e  ", iter, f, ∇fNorm2, get_property(tr, :radius))

    optimal = ∇fNorm2 ≤ ϵ
    elapsed_time = time() - start_time
    tired = neval_residual(nlp) > max_f || elapsed_time > max_time
  end
  @info infoline

  if optimal
    status = :first_order
  elseif tired
    if neval_residual(nlp) > max_f
      status = :max_eval
    elseif elapsed_time > max_time
      status = :max_time
    end
  end

  return GenericExecutionStats(status, nlp, solution=x, objective=f, dual_feas=∇fNorm2,
                               iter=iter, elapsed_time=elapsed_time)
end
