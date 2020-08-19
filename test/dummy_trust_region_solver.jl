using Krylov

function dummy_trust_region_solver(
  nlp :: AbstractNLPModel;
  x :: AbstractVector = copy(nlp.meta.x0),
  atol :: Real = 1e-6,
  rtol :: Real = 1e-6,
  max_eval :: Int = 1000,
  max_time :: Float64 = 30.0,
  tr_method :: Symbol = :basic,
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
  ϕ = merit_constructor(nlp, 1e-3, cx=copy(cx), gx=gx)

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
    if ncon > 0
      # Normal step
      v = cgls(Jx, -cx, radius=0.8Δ)[1]
      # Tangent step
      Z = nullspace(Jx)
      Δx = v + Z * cg(Z' * Hxy * Z, -Z' * (dual + Hxy * v), radius=sqrt(Δ^2 - dot(v, v)))[1]
    else
      Δx = cg(Hxy, -dual, radius=Δ)[1]
    end

    Δq = dot(Δx, gx) + dot(Δx, Hxy * Δx) / 2
    Ad = Jx * Δx
    tro = trust_region!(ϕ, x, Δx, xt, Δq, Ad, Δ, method=tr_method, update_obj_at_x=false)

    Δ = tro.Δ
    x .= tro.xt

    if tro.success
      fx = ϕ.fx
      cx .= ϕ.cx
      grad!(nlp, x, gx)
      if ncon > 0
        Jx = jac(nlp, x)
      end
      y = cgls(Jx', -gx)[1]
      dual = gx + Jx' * y
    else
      ϕ.fx = fx
      ϕ.cx = cx
    end

    elapsed_time = time() - start_time
    solved = norm(dual) < ϵd && norm(cx) < ϵp
    tired = evals(nlp) > max_eval || elapsed_time > max_time

    iter += 1
    @info log_row(Any[iter, evals(nlp), fx, norm(cx), norm(dual), elapsed_time, Δ, ϕ.η, tro.status])
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
