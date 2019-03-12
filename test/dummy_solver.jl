function dummy_solver(nlp :: AbstractNLPModel;
                      x :: AbstractVector = nlp.meta.x0,
                      atol :: Real = sqrt(eps(eltype(x))),
                      rtol :: Real = sqrt(eps(eltype(x))),
                      max_eval :: Int = 1000,
                      max_time :: Float64 = 30.0,
                     )

  start_time = time()
  elapsed_time = 0.0

  nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
  T = eltype(x)

  fx = obj(nlp, x)
  cx = ncon > 0 ? cons(nlp, x) : zeros(T)
  α = one(T)

  iter = 0
  @info log_header([:iter, :f, :c, :t], [Int, T, T, Float64])
  @info log_row(Any[iter, fx, norm(cx), elapsed_time])
  tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || elapsed_time > max_time

  while !tired
    xt = x + α * convert.(T, randn(nvar))
    ft = obj(nlp, xt)
    ct = ncon > 0 ? cons(nlp, xt) : zeros(T)
    if ft < fx && norm(ct) ≤ norm(cx)
      x .= xt
      fx = ft
      cx .= ct
    else
      α = 9α / 10
    end

    iter += 1
    elapsed_time = time() - start_time
    tired = neval_obj(nlp) + neval_cons(nlp) > max_eval || elapsed_time > max_time
    @info log_row(Any[iter, fx, norm(cx), elapsed_time])
  end

  return GenericExecutionStats(:unknown, nlp,
                               objective=fx,
                               dual_feas=norm(grad(nlp, x)),
                               primal_feas=norm(cx),
                               elapsed_time=elapsed_time,
                               solution=x,
                              )
end
