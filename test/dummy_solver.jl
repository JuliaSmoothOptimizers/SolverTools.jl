function dummy_solver(
  nlp::AbstractNLPModel;
  x::AbstractVector = nlp.meta.x0,
  atol::Real = sqrt(eps(eltype(x))),
  rtol::Real = sqrt(eps(eltype(x))),
  max_eval::Int = 1000,
  max_time::Float64 = 30.0,
)
  start_time = time()
  elapsed_time = 0.0

  nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
  T = eltype(x)

  cx = ncon > 0 ? cons(nlp, x) : zeros(T, 0)
  gx = grad(nlp, x)
  Jx = ncon > 0 ? jac(nlp, x) : zeros(T, 0, nvar)
  y = -Jx' \ gx
  Hxy = ncon > 0 ? hess(nlp, x, y) : hess(nlp, x)

  dual = gx + Jx' * y

  iter = 0

  ϵd = atol + rtol * norm(dual)
  ϵp = atol

  fx = obj(nlp, x)
  @info log_header([:iter, :f, :c, :dual, :t], [Int, T, T, Float64])
  @info log_row(Any[iter, fx, norm(cx), norm(dual), elapsed_time])
  solved = norm(dual) < ϵd && norm(cx) < ϵp
  tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || elapsed_time > max_time

  while !(solved || tired)
    Hxy = ncon > 0 ? hess(nlp, x, y) : hess(nlp, x)
    W = Symmetric([Hxy zeros(T, nvar, ncon); Jx zeros(T, ncon, ncon)], :L)
    Δxy = -W \ [dual; cx]
    Δx = Δxy[1:nvar]
    Δy = Δxy[(nvar + 1):end]
    x += Δx
    y += Δy

    cx = ncon > 0 ? cons(nlp, x) : zeros(T, 0)
    gx = grad(nlp, x)
    Jx = ncon > 0 ? jac(nlp, x) : zeros(T, 0, nvar)
    dual = gx + Jx' * y
    elapsed_time = time() - start_time
    solved = norm(dual) < ϵd && norm(cx) < ϵp
    tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || elapsed_time > max_time

    iter += 1
    fx = obj(nlp, x)
    @info log_row(Any[iter, fx, norm(cx), norm(dual), elapsed_time])
  end

  status = if solved
    :first_order
  elseif elapsed_time > max_time
    :max_time
  else
    :max_eval
  end

  return GenericExecutionStats(
    :unknown,
    nlp,
    objective = fx,
    dual_feas = norm(dual),
    primal_feas = norm(cx),
    multipliers = y,
    multipliers_L = zeros(T, nvar),
    multipliers_U = zeros(T, nvar),
    elapsed_time = elapsed_time,
    solution = x,
    iter = iter,
  )
end
