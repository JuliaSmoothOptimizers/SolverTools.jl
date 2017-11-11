export display_header, solve_problems, solve_problem

type SkipException <: Exception
end

optimizelogger = get_logger("optimize")


"""
    display_header()

Output header for stats table.

This function is called once before the first problem solve and can be overridden to customize the display.

#### Return value
Nothing.
"""
function display_header()
  @info(optimizelogger,
        @sprintf("%-15s  %8s  %9s  %7s  %5s  %5s  %6s  %s",
                 "Name", "nvar", "f", "‖∇f‖", "#obj", "#grad", "#hprod", "status"))
end


"""
    display_problem_stats(nlp, f, gNorm, status)

Output stats for problem `nlp` after a solve.

This function is called after each problem solve and can be overridden to customize the display.

#### Arguments
* `nlp::AbstractNLPModel`: the problem just solved
* `f::Float64`: final objective value
* `gNorm::Float64`: final gradient norm
* `status::String`: final solver status or error message.

#### Return value
Nothing.
"""
function display_problem_stats(nlp::AbstractNLPModel, f::Float64, gNorm::Float64, status::String)
  @info(optimizelogger,
        @sprintf("%-15s  %8d  %9.2e  %7.1e  %5d  %5d  %6d  %s",
                 nlp.meta.name, nlp.meta.nvar, f, gNorm,
                 nlp.counters.neval_obj, nlp.counters.neval_grad,
                 nlp.counters.neval_hprod, status))
end


"""
    solve_problems(solver :: Function, problems :: Any; kwargs...)

Apply a solver to a set of problems.

#### Arguments
* `solver`: the function name of a solver
* `problems`: the set of problems to pass to the solver, as an iterable of `AbstractNLPModel`
  (it is recommended to use a generator expression)

#### Keyword arguments
* `prune`: do not include skipped problems in the final statistics (default: `true`)
* any other keyword argument accepted by `run_problem()`

#### Return value
* an `Array{Int}(nprobs, 3)` where `nprobs` is the number of problems in the problem.
  See the documentation of `solve_problem()` for the form of each entry.
"""
function solve_problems(solver :: Function, problems :: Any; prune :: Bool=true, kwargs...)
  display_header()
  nprobs = length(problems)
  solverstr = split(string(solver), ".")[end]
  solverlogger = get_logger("optimize.$(solverstr)")
  current_level = solverlogger.level
  solverlogger.level = nprobs > 1 ? MiniLogging.WARN : MiniLogging.INFO
  stats = -ones(nprobs, 3)
  k = 0
  for problem in problems
    try
      (f, g, h) = solve_problem(solver, problem; kwargs...)
      k = k + 1
      stats[k, :] = [f, g, h]
      finalize(problem)
    catch e
      isa(e, SkipException) || rethrow(e)
    end
  end
  solverlogger.level = current_level
  return prune ? stats[1:k, :] : stats
end


"""
    solve_problem(solver :: Function, nlp :: AbstractNLPModel; kwargs...)

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
function solve_problem(solver :: Function, nlp :: AbstractNLPModel; kwargs...)
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
    status = :msg in fieldnames(e) ? e.msg : string(e)
  end
  # if nlp.scale_obj
  #   f /= nlp.scale_obj_factor
  #   gNorm /= nlp.scale_obj_factor
  # end
  display_problem_stats(nlp, f, gNorm, status)
  return optimal ? (nlp.counters.neval_obj, nlp.counters.neval_grad, nlp.counters.neval_hprod) : (-nlp.counters.neval_obj, -nlp.counters.neval_grad, -nlp.counters.neval_hprod)
end
