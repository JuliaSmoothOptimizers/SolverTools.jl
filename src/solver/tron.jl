#  Some parts of this code were adapted from
# https://github.com/PythonOptimizers/NLP.py/blob/develop/nlp/optimize/tron.py

using LinearOperators, NLPModels

export tron

"""`tron(nlp)`

A pure Julia implementation of a trust-region solver for bound-constrained
optimization:

    min f(x)    s.t.    ℓ ≦ x ≦ u

TRON is described in

Chih-Jen Lin and Jorge J. Moré, *Newton's Method for Large Bound-Constrained
Optimization Problems*, SIAM J. Optim., 9(4), 1100–1127, 1999.
"""
function tron(nlp :: AbstractNLPModel;
              μ₀ :: Real=1e-2,
              μ₁ :: Real=1.0,
              σ :: Real=10.0,
              itmax :: Int=10_000 * nlp.meta.nvar,
              max_cgiter :: Int=nlp.meta.nvar,
              cgtol :: Real=0.1,
              timemax :: Real=60.0,
              atol :: Real=1e-8,
              rtol :: Real=1e-6,
              fatol :: Real=0.0,
              frtol :: Real=1e-12
             )
  ℓ = nlp.meta.lvar
  u = nlp.meta.uvar
  f(x) = obj(nlp, x)
  g(x) = grad(nlp, x)
  n = nlp.meta.nvar

  iter = 0
  start_time = time()
  el_time = 0.0

  # Preallocation
  temp = zeros(n)
  gpx = zeros(n)
  xc = zeros(n)
  Hs = zeros(n)

  x = max.(ℓ, min.(nlp.meta.x0, u))
  fx = f(x)
  gx = g(x)
  num_success_iters = 0

  # Optimality measure
  project_step!(gpx, x, gx, ℓ, u, -1.0)
  πx = norm(gpx)
  ϵ = atol + rtol * πx
  fmin = min(-1.0, fx) / eps(eltype(x))
  optimal = πx <= ϵ
  tired = iter >= itmax || el_time > timemax
  unbounded = fx < fmin
  stalled = false
  status = :unknown

  αC = 1.0
  tr = TRONTrustRegion(min(max(1.0, 0.1 * norm(πx)), 100))
  @info @sprintf("%4s  %9s  %7s  %7s  %7s  %s",
                 "Iter", "f", "π", "Radius", "Ratio", "CG-status")
  @info @sprintf("%4d  %9.2e  %7.1e  %7.1e",
                 iter, fx, πx, get_property(tr, :radius))
  while !(optimal || tired || stalled || unbounded)
    # Current iteration
    xc .= x
    fc = fx
    Δ = get_property(tr, :radius)
    H = hess_op!(nlp, xc, temp)

    αC, s, cauchy_status = cauchy(x, H, gx, Δ, αC, ℓ, u, μ₀=μ₀, μ₁=μ₁, σ=σ)

    if cauchy_status != :success
      @error "Cauchy step returned: $cauchy_status"
      status = cauchy_status
      stalled = true
      continue
    end
    s, Hs, cgits, cginfo = projected_newton!(x, H, gx, Δ, cgtol, s, ℓ, u, max_cgiter=max_cgiter)
    slope = dot(gx, s)
    qs = 0.5 * dot(s, Hs) + slope
    fx = f(x)

    ared, pred, quad_min = aredpred(tr, nlp, fc, fx, qs, x, s, slope)
    if pred ≥ 0
      status = :neg_pred
      stalled = true
      continue
    end
    tr.ratio = ared / pred
    tr.quad_min = quad_min

    s_norm = norm(s)
    if num_success_iters == 0
      tr.radius = min(Δ, s_norm)
    end

    # Update the trust region
    update!(tr, s_norm)

    if acceptable(tr)
      num_success_iters += 1
      gx = g(x)
      project_step!(gpx, x, gx, ℓ, u, -1.0)
      πx = norm(gpx)
    end

    # No post-iteration

    if !acceptable(tr)
      fx = fc
      x .= xc
    end

    iter += 1
    el_time = time() - start_time
    tired = iter >= itmax ||  el_time > timemax
    optimal = πx <= ϵ
    unbounded = fx < fmin

    @info @sprintf("%4d  %9.2e  %7.1e  %7.1e  %7.1e  %s",
                   iter, fx, πx, Δ, tr.ratio, cginfo)
  end

  if tired
    status = iter >= itmax ? :max_iter : :max_time
  elseif optimal
    status = :first_order
  elseif unbounded
    status = :unbounded
  end

  return GenericExecutionStats(status, nlp, solution=x, objective=fx, dual_feas=πx,
                               iter=iter, elapsed_time=el_time)
end

"""`s = projected_line_search!(x, H, g, d, ℓ, u; μ₀ = 1e-2)`

Performs a projected line search, searching for a step size `t` such that

    0.5sᵀHs + sᵀg ≦ μ₀sᵀg,

where `s = P(x + t * d) - x`, while remaining on the same face as `x + d`.
Backtracking is performed from t = 1.0. `x` is updated in place.
"""
function projected_line_search!(x::AbstractVector{T},
                                H::Union{AbstractMatrix,AbstractLinearOperator},
                                g::AbstractVector{T},
                                d::AbstractVector{T},
                                ℓ::AbstractVector{T},
                                u::AbstractVector{T}; μ₀::Real = 1e-2) where T <: Real
  α = one(T)
  _, brkmin, _ = breakpoints(x, d, ℓ, u)
  nsteps = 0
  n = length(x)

  s = zeros(n)
  Hs = zeros(n)

  search = true
  while search && α > brkmin
    nsteps += 1
    project_step!(s, x, d, ℓ, u, α)
    slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)
    if qs <= μ₀ * slope
      search = false
    else
      α /= 2
    end
  end
  if α < 1.0 && α < brkmin
    α = brkmin
    project_step!(s, x, d, ℓ, u, α)
    slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)
  end

  project_step!(s, x, d, ℓ, u, α)
  x .= x .+ s

  return s
end

"""`α, s = cauchy(x, H, g, Δ, ℓ, u; μ₀ = 1e-2, μ₁ = 1.0, σ=10.0)`

Computes a Cauchy step `s = P(x - α g) - x` for

    min  q(s) = ¹/₂sᵀHs + gᵀs     s.t.    ‖s‖ ≦ μ₁Δ,  ℓ ≦ x + s ≦ u,

with the sufficient decrease condition

    q(s) ≦ μ₀sᵀg.
"""
function cauchy(x::AbstractVector{T},
                H::Union{AbstractMatrix,AbstractLinearOperator},
                g::AbstractVector{T},
                Δ::Real, α::Real, ℓ::AbstractVector{T}, u::AbstractVector{T};
                μ₀::Real = 1e-2, μ₁::Real = 1.0, σ::Real = 10.0) where T <: Real
  # TODO: Use brkmin to care for g direction
  _, _, brkmax = breakpoints(x, -g, ℓ, u)
  n = length(x)
  s = zeros(n)
  Hs = zeros(n)

  status = :success

  project_step!(s, x, g, ℓ, u, -α)

  # Interpolate or extrapolate
  s_norm = norm(s)
  if s_norm > μ₁ * Δ
    interp = true
  else
    slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)
    interp = qs >= μ₀ * slope
  end

  if interp
    search = true
    while search
      α /= σ
      project_step!(s, x, g, ℓ, u, -α)
      s_norm = norm(s)
      if s_norm <= μ₁ * Δ
        slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)
        search = qs >= μ₀ * slope
      end
      # TODO: Correctly assess why this fails
      if α < sqrt(nextfloat(zero(α)))
        stalled = true
        search = false
        status = :small_step
      end
    end
  else
    search = true
    αs = α
    while search && α <= brkmax
      α *= σ
      project_step!(s, x, g, ℓ, u, -α)
      s_norm = norm(s)
      if s_norm <= μ₁ * Δ
        slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)
        if qs <= μ₀ * slope
          αs = α
        end
      else
        search = false
      end
    end
    # Recover the last successful step
    α = αs
    s = project_step!(s, x, g, ℓ, u, -α)
  end
  return α, s, status
end

"""`projected_newton!(x, H, g, Δ, gctol, s, max_cgiter, ℓ, u)`

Compute an approximate solution `d` for

min q(d) = ¹/₂dᵀHs + dᵀg    s.t.    ℓ ≦ x + d ≦ u,  ‖d‖ ≦ Δ

starting from `s`.  The steps are computed using the conjugate gradient method
projected on the active bounds.
"""
function projected_newton!(x::AbstractVector{T}, H::Union{AbstractMatrix,AbstractLinearOperator},
                           g::AbstractVector{T}, Δ::Real, cgtol::Real, s::AbstractVector{T},
                           ℓ::AbstractVector{T}, u::AbstractVector{T};
                           max_cgiter::Int = max(50, length(x))) where T <: Real
  n = length(x)
  status = ""

  Hs = H * s

  # Projected Newton Step
  exit_optimal = false
  exit_pcg = false
  exit_itmax = false
  iters = 0
  x .= x .+ s
  project!(x, x, ℓ, u)
  while !(exit_optimal || exit_pcg || exit_itmax)
    ifree = setdiff(1:n, active(x, ℓ, u))
    if length(ifree) == 0
      exit_optimal = true
      continue
    end
    Z = opExtension(ifree, n)
    @views wa = g[ifree]
    @views gfree = Hs[ifree] + wa
    gfnorm = norm(wa)

    ZHZ = Z' * H * Z
    st, stats = Krylov.cg(ZHZ, -gfree, radius=Δ, rtol=cgtol, atol=0.0,
                          itmax=max_cgiter)
    iters += length(stats.residuals)
    status = stats.status

    # Projected line search
    xfree = x[ifree]
    @views w = projected_line_search!(xfree, ZHZ, gfree, st, ℓ[ifree], u[ifree])
    @views s[ifree] .+= w

    Hs .= H * s

    @views gfree .= Hs[ifree] .+ g[ifree]
    if norm(gfree) <= cgtol * gfnorm
      exit_optimal = true
    elseif status == "on trust-region boundary"
      exit_pcg = true
    elseif iters >= max_cgiter
      exit_itmax = true
    end
  end
  status = if exit_optimal
    "stationary point found"
  elseif exit_itmax
    "maximum number of iterations"
  else
    status # on trust-region
  end

  return s, Hs, iters, status
end
