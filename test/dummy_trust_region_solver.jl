using Krylov

function dummy_trust_region_solver(
  nlp :: AbstractNLPModel;
  x :: AbstractVector = copy(nlp.meta.x0),
  atol :: Real = 1e-6,
  rtol :: Real = 1e-6,
  max_eval :: Int = 1000,
  max_time :: Float64 = 30.0,
  # ls_method :: Symbol = :armijo,
  merit_constructor = L1Merit
)

  start_time = time()
  elapsed_time = 0.0
  evals(nlp) = neval_obj(nlp) + neval_cons(nlp)

  nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
  T = eltype(x)
  xt = copy(x)

  cx = ncon > 0 ? cons(nlp, x) : zeros(T, 0)
  gx = grad(nlp, x)
  Jx = ncon > 0 ? jac(nlp, x) : zeros(T, 0, nvar)
  y = cgls(Jx', -gx)[1]

  dual = gx + Jx' * y

  iter = 0

  Δ = 100.0
  ϕ = merit_constructor(nlp, 1e-3, cx=cx, gx=gx)

  ϵd = atol + rtol * norm(dual)
  ϵp = atol

  fx = obj(nlp, x)
  ϕ.fx = fx
  @info log_header([:iter, :nfc, :f, :c, :dual, :time, :Δ, :η, :iterkind], [Int, Int, T, T, T, T, T, T, String])
  @info log_row(Any[iter, evals(nlp), fx, norm(cx), norm(dual), elapsed_time, Δ, ϕ.η, "Initial"])
  solved = norm(dual) < ϵd && norm(cx) < ϵp
  tired = evals(nlp) > max_eval || elapsed_time > max_time

  while !(solved || tired)
    Hxy = ncon > 0 ? hess_op(nlp, x, y) : hess_op(nlp, x)
    local Δx
    ϕx = obj(ϕ, x, update=false)
    ϕxprimal = primalobj(ϕ)
    ϕxdual   = dualobj(ϕ)
    if ncon > 0
      # Normal step
      v = cgls(Jx, -cx, radius=0.8Δ)[1]
      # Tangent step
      Z = nullspace(Jx)
      Δx = v + Z * cg(Z' * Hxy * Z, -Z' * (dual + Hxy * v), radius=sqrt(Δ^2 - dot(v, v)))[1]
    else
      Δx = cg(Hxy, -dual, radius=Δ)[1]
    end

    # Workaround to be agnostic to the merit function
    # Pretend fx is the quadratic approximation to obj(nlp, x + Δx)
    # Pretend cx is que linear approximation to cons(nlp, x + Δx)
    ϕ.fx += dot(Δx, gx) + dot(Δx, Hxy * Δx) / 2
    Ad = Jx * Δx
    cpAd = cx .+ Ad
    ct = ϕ.cx
    ϕ.cx = cpAd
    ϕtprimal = primalobj(ϕ)
    ϕtdual = dualobj(ϕ)

    xt = x + Δx
    ϕt = obj(ϕ, xt, update=true)
    ft = ϕ.fx

    @assert ϕxprimal ≥ ϕtprimal
    while ϕxdual - ϕtdual < -0.1 * ϕ.η * (ϕxprimal - ϕtprimal) < 0 # For unconstrained problems, right side is 0.
      ϕ.η *= 2
    end

    Ared = ϕx - ϕt
    Pred = ϕxdual - ϕtdual + ϕ.η * (ϕxprimal - ϕtprimal)

    ρ = Ared / Pred
    iter_kind = if ρ > 1e-2 # accept
      x .= xt
      fx = ft
      cx = ϕ.cx
      grad!(nlp, x, gx) # Updates ϕ.gx
      if ncon > 0
        Jx = jac(nlp, x)
      end
      y = cgls(Jx', -gx)[1]
      dual = gx + Jx' * y
      if ρ > 0.75 && norm(Δx) > 0.9Δ
        Δ *= 2
        :great
      else
        :good
      end
    else
      ϕ.fx = fx
      ϕ.cx = cx
      Δ /= 4
      :bad
    end

    elapsed_time = time() - start_time
    solved = norm(dual) < ϵd && norm(cx) < ϵp
    tired = evals(nlp) > max_eval || elapsed_time > max_time

    iter += 1
    @info log_row(Any[iter, evals(nlp), fx, norm(cx), norm(dual), elapsed_time, Δ, ϕ.η, iter_kind])
  end

  status = if solved
    :first_order
  elseif elapsed_time > max_time
    :max_time
  else
    :max_eval
  end

  return GenericExecutionStats(status, nlp,
                               objective=fx, dual_feas=norm(dual), primal_feas=norm(cx),
                               multipliers=y, multipliers_L=zeros(T, nvar), multipliers_U=zeros(T, nvar),
                               elapsed_time=elapsed_time, solution=x, iter=iter
                              )
end
