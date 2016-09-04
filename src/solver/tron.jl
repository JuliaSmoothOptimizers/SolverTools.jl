using LinearOperators
# A trust-region solver for unconstrained optimization with bounds

export tron

function active(x, l, u; ϵlu = 1e-4)
  if isa(ϵlu, Float64)
    ϵlu = (u-l)*ϵlu
  end
  A = Int[]
  W = Int[]
  n = length(x)
  for i = 1:n
    if x[i] < l[i] + ϵlu[i]
      push!(A, i)
      push!(W, 0)
    elseif x[i] > u[i] - ϵlu[i]
      push!(A, i)
      push!(W, 1)
    end
  end
  return A, W
end

function tron(nlp :: AbstractNLPModel; bfgs :: Bool=true, μ₀ :: Real=1e-2,
    μ₁ :: Real=1.0, σ :: Real=1.2, verbose=false, itmax :: Integer=100000,
    timemax :: Real=60, mem :: Integer=5, atol :: Real=1e-8, rtol :: Real=1e-6)
  l = nlp.meta.lvar
  u = nlp.meta.uvar
  f(x) = obj(nlp, x)
  g(x) = grad(nlp, x)
  n = nlp.meta.nvar
  if bfgs
    if n <= 5
      mem = n
    end
    H = LBFGSOperator(n, mem, scaling = true)
  end

  iter = 0
  start_time = time()
  el_time = 0.0

  # Projection
  ξ(x) = max(min(x, u), l)
  P!(y, x, a, v) = begin # y = P[x + a*v] - x, where l ≦ x ≦ u
    for i = 1:n
      y[i] = a*v[i] > 0 ? min(x[i]+a*v[i], u[i])-x[i] : max(x[i]+a*v[i], l[i])-x[i]
    end
    return y
  end

  # Preallocation
  xcur = zeros(n)
  dcur = zeros(n)
  sα = zeros(n)
  sαn = zeros(n)
  sαp = zeros(n)
  wβ = zeros(n)
  gpx = zeros(n)

  x = ξ(nlp.meta.x0)
  gx = g(x)
  ϵ = atol + rtol * norm(gx)

  # Optimality measure
  P!(gpx, x, -1.0, gx)
  πx = norm(gpx)
  optimal = πx <= ϵ
  tired = iter >= itmax ||  el_time > timemax
  stalled = false

  α = 1.0
  fx = f(x)
  tr = TrustRegion(min(max(0.1 * norm(gx), 1.0), 100.0))
  Δ() = get_property(tr, :radius)
  ρ = Inf
  if verbose
    @printf("%4s  %9s  %7s  %7s  %8s\n", "Iter", "f", "π", "Radius", "Ratio")
    @printf("%4d  %9.2e  %7.1e  %7.1e\n", iter, fx, πx, Δ())
  end
  while !(optimal || tired || stalled)
    # Model
    if !bfgs
      H = LinearOperator(n, Float64, v -> hprod(nlp, x, v))
    end
    q(d) = 0.5*dot(d, H * d) + dot(d, gx)
    # Projected step
    P!(sα, x, -α, gx)
    copy!(sαp, sα)

    # Find α satisfying the decrease condition increasing if it's
    # possible, or decreasing if necessary.
    if q(sα) <= μ₀*dot(gx,sα) && norm(sα) <= μ₁*Δ()
      while q(sα) <= μ₀*dot(gx,sα) && norm(sα) <= μ₁*Δ()
        α *= σ
        P!(sαn, x, -α, gx)
        # Check if the step is in a corner.
        sαn == sα && break
        copy!(sαp, sα)
        copy!(sα, sαn)
        if α > 1e12
          break
        end
      end
      copy!(sα, sαp)
      α /= σ
    else
      while q(sα) > μ₀*dot(gx,sα) || norm(sα) > μ₁*Δ()
        α /= σ
        copy!(sαp, sα)
        P!(sα, x, -α, gx)
        if α < 1e-24
          stalled = true
          break
        end
      end
    end

    stalled && break

    Δcur = norm(sα)
    copy!(dcur, sα)
    copy!(xcur, x)
    BLAS.axpy!(1.0, sα, xcur)

    nsmall = 0

    while Δcur < Δ()
      A, W = active(xcur, l, u)
      I = setdiff(1:n, A)
      if length(I) == 0
        break
      end
      v = H * dcur + gx
      if norm(v[I]) < 1e-5
        break
      end
      st, st_status = steihaug(H, v, Δ()-Δcur, I = I, ϵ=ϵ, kmax=n)
      β = 1.0
      P!(wβ, xcur, β, st)
      if norm(wβ) < 1e-5
        break
      end
      qdcur = q(dcur)
      while q(dcur + wβ) > qdcur + μ₀*min(dot(v, wβ), 0)
        β *= 0.9
        P!(wβ, xcur, β, st)
        if norm(wβ) < 1e-5
          break
        end
      end
      if norm(wβ) < 1e-5
        break
      end
      Δcur += norm(wβ)
      BLAS.axpy!(1.0, wβ, dcur)
      BLAS.axpy!(1.0, wβ, xcur)
      if norm(wβ) < 1e-3*Δ()
        nsmall += 1
        if nsmall == 3
          break
        end
      end
    end

    # Candidate
    fxcur = f(xcur)

    # Ratio
    try
      ρ = ratio(fx, fxcur, q(dcur))
    catch e
      # Failed
      status = e.msg
      stalled = true
      break
    end

    # Update x
    if acceptable(tr, ρ)
      copy!(x, xcur)
      fx = fxcur

      y = gx
      gx = g(x)
      P!(gpx, x, -1.0, gx)
      πx = norm(gpx)
      if bfgs # Update bfgs
        push!(H, dcur, gx - y)
      end
    end

    # Update the trust region
    update!(tr, ρ, norm(sα))

    iter += 1
    el_time = time() - start_time
    tired = iter >= itmax ||  el_time > timemax
    optimal = πx <= ϵ

    verbose && @printf("%4d  %9.2e  %7.1e  %7.1e  %8.1e\n", iter, fx, πx, Δ(), ρ)
  end

  if tired
    status = iter >= itmax ? "maximum number of iterations" : "maximum elapsed time"
  elseif !stalled
    status = "first-order stationary"
  end

  return x, f(x), πx, iter, optimal, tired, status, el_time
end

