export display_header, run_problems, run_mpb_problem, run_ampl_problem, run_solver

type SkipException <: Exception
end

function display_header()
  @printf("%-15s  %8s  %9s  %7s  %5s  %5s  %6s  %s\n",
          "Name", "nvar", "f", "‖∇f‖", "#obj", "#grad", "#hprod", "status")
end


"""
    run_problems(solver :: Function, problems :: Any; kwargs...)

Apply a solver to a set of problems.

#### Arguments
* `solver`: the function name of a solver
* `problems`: the set of problems to pass to the solver, as a list of symbols
* `dim`: the approximate size in which each problem should be instantiated.
  The problem size may be adjusted automatically to the nearest smaller size
  if a particular problem's size is constrained.
  This argument has no effect on certain problem formats (see `format` below).

#### Keyword arguments
* `format` the problem format. Currently, only `:mpb` and `:ampl` are supported
* any other keyword argument accepted by `run_problem()`

#### Return value
* an `Array(Int, nprobs, 3)` where `nprobs` is the number of problems in the problem.
  See the documentation of `run_solver()` for the form of each entry.
"""
function run_problems(solver :: Function, problems :: Any; kwargs...)
  display_header()
  nprobs = length(problems)
  verbose = nprobs ≤ 1
  stats = -ones(nprobs, 3)
  k = 1
  for problem in problems
    try
      (f, g, h) = run_solver(solver, problem, verbose=verbose; kwargs...)
      stats[k, :] = [f, g, h]
      k = k + 1
    catch e
      isa(e, SkipException) || rethrow(e)
    end
  end
  return stats
end


"""
    run_solver(solver :: Function, nlp :: AbstractNLPModel; kwargs...)

Apply a solver to a generic `AbstractNLPModel`.

#### Arguments
* `solver`: the function name of a solver, as a symbol
* `nlp`: an `AbstractNLPModel` instance

#### Keyword arguments
Any keyword argument accepted by the solver.

#### Return value
* an array `[f, g, h]` representing the number of objective evaluations, the number
  of gradient evaluations and the number of Hessian-vector products required to solve
  `nlp` with `solver`.
  Negative values are used to represent failures.
"""
function run_solver(solver :: Function, nlp :: AbstractNLPModel; kwargs...)
  args = Dict(kwargs)
  skip = haskey(args, :skipif) ? pop!(args, :skipif) : x -> false
  skip(nlp) && throw(SkipException())

  # Julia nonsense
  optimal = false
  f = 0.0
  gNorm = 0.0
  status = "fail"
  try
    (x, f, gNorm, iter, optimal, tired, status) = solver(nlp; args...)
  catch e
    println(e)
    status = e.msg
  end
  # if nlp.scale_obj
  #   f /= nlp.scale_obj_factor
  #   gNorm /= nlp.scale_obj_factor
  # end
  @printf("%-15s  %8d  %9.2e  %7.1e  %5d  %5d  %6d  %s\n",
          nlp.meta.name, nlp.meta.nvar, f, gNorm,
          nlp.counters.neval_obj, nlp.counters.neval_grad,
          nlp.counters.neval_hprod, status)
  return optimal ? (nlp.counters.neval_obj, nlp.counters.neval_grad, nlp.counters.neval_hprod) : (-nlp.counters.neval_obj, -nlp.counters.neval_grad, -nlp.counters.neval_hprod)
end
