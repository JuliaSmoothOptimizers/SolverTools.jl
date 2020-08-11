function dummy_linesearch_solver(
  nlp :: AbstractNLPModel;
  x :: AbstractVector = nlp.meta.x0,
  atol :: Real = 1e-6,
  rtol :: Real = 1e-6,
  max_eval :: Int = 1000,
  max_time :: Float64 = 30.0,
  ls_method :: Symbol = :armijo,
  merit_constructor = L1Merit
)

  start_time = time()
  elapsed_time = 0.0

  nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
  T = eltype(x)
  xt = copy(x)

  cx = ncon > 0 ? cons(nlp, x) : zeros(T, 0)
  gx = grad(nlp, x)
  Jx = ncon > 0 ? jac(nlp, x) : zeros(T, 0, nvar)
  y = -Jx' \ gx
  Hxy = ncon > 0 ? hess(nlp, x, y) : hess(nlp, x)

  dual = gx + Jx' * y

  iter = 0

  ϕ = merit_constructor(nlp, 1e-3, cx=cx, gx=gx)

  ϵd = atol + rtol * norm(dual)
  ϵp = atol

  fx = obj(nlp, x)
  @info log_header([:iter, :f, :c, :dual, :time, :η], [Int, T, T, T, T, T])
  @info log_row(Any[iter, fx, norm(cx), norm(dual), elapsed_time, ϕ.η])
  solved = norm(dual) < ϵd && norm(cx) < ϵp
  tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || elapsed_time > max_time

  while !(solved || tired)
    Hxy = ncon > 0 ? hess(nlp, x, y) : hess(nlp, x)
    W = Symmetric([Hxy  zeros(T, nvar, ncon); Jx  zeros(T, ncon, ncon)], :L)
    Δxy = -W \ [dual; cx]
    Δx = Δxy[1:nvar]
    Δy = Δxy[nvar+1:end]

    ϕ.fx = fx
    ncon > 0 && jprod!(nlp, x, Δx, ϕ.Ad)
    Dϕx = derivative(ϕ, x, Δx, update=false)
    while Dϕx ≥ 0
      ϕ.η *= 2
      Dϕx = derivative(ϕ, x, Δx, update=false)
    end
    ϕx = obj(ϕ, x, update=false)

    lso = linesearch!(ϕ, x, Δx, xt, method=ls_method)
    x .= xt
    t = lso.t
    y .+= t * Δy

    grad!(nlp, x, gx) # Updates ϕ.gx
    if ncon > 0
      Jx = jac(nlp, x)
    end
    dual = gx + Jx' * y
    elapsed_time = time() - start_time
    solved = norm(dual) < ϵd && norm(cx) < ϵp
    tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || elapsed_time > max_time

    iter += 1
    fx = obj(nlp, x)
    @info log_row(Any[iter, fx, norm(cx), norm(dual), elapsed_time, ϕ.η])
  end

  status = if solved
    :first_order
  elseif elapsed_time > max_time
    :max_time
  else
    :max_eval
  end

  return GenericExecutionStats(:unknown, nlp,
                               objective=fx, dual_feas=norm(dual), primal_feas=norm(cx),
                               multipliers=y, multipliers_L=zeros(T, nvar), multipliers_U=zeros(T, nvar),
                               elapsed_time=elapsed_time, solution=x, iter=iter
                              )
end
