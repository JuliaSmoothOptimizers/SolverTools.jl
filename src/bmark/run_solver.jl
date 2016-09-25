export display_header, run_problems, run_mpb_problem, run_ampl_problem, run_solver

type SkipException <: Exception
end

function display_header()
  @printf("%-15s  %8s  %9s  %7s  %5s  %5s  %6s  %s\n",
          "Name", "nvar", "f", "‖∇f‖", "#obj", "#grad", "#hprod", "status")
end


"""
    run_problems(solver :: Symbol, problems :: Vector{Symbol}, dim :: Int; kwargs...)

Apply a solver to a set of problems.

#### Arguments
* `solver`: the function name of a solver, as a symbol
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
function run_problems(solver :: Symbol, problems :: Vector{Symbol}, dim :: Int; format :: Symbol=:mpb, args...)
  format in (:mpb, :ampl) || error("format not recognized---use :mpb or :ampl")
  run_problem = eval(Symbol("run_" * string(format) * "_problem"))
  display_header()
  nprobs = length(problems)
  verbose = nprobs ≤ 1
  stats = -ones(nprobs, 3)
  k = 1
  for problem in problems
    try
      (f, g, h) = run_problem(solver, problem, dim, verbose=verbose; args...)
      stats[k, :] = [f, g, h]
      k = k + 1
    catch e
      isa(e, SkipException) || rethrow(e)
    end
  end
  return stats
end


"""
    run_mpb_problem(solver :: Symbol, problem :: Symbol, dim :: Int; kwargs...)

Apply a solver to a problem as a `MathProgNLPModel`.

#### Arguments
See the documentation of `run_problems()`.

#### Keyword arguments
Any keyword argument accepted by `run_solver()`.

#### Return value
See the documentation of `run_solver()`.
"""
function run_mpb_problem(solver :: Symbol, problem :: Symbol, dim :: Int; args...)
  problem_f = eval(problem)
  nlp = MathProgNLPModel(problem_f(dim), name=string(problem))
  # scale_obj!(nlp)  # not implemented
  stats = run_solver(solver, nlp; args...)
  # unscale_obj!(nlp)  # not implemented
  return stats
end


"""
    run_ampl_problem(solver :: Symbol, problem :: Symbol, dim :: Int; kwargs...)

Apply a solver to a problem as an `AmplModel`.

#### Arguments
See the documentation of `run_problems()`.

#### Keyword arguments
* any keyword argument accepted by `run_solver()`

#### Return value
See the documentation of `run_solver()`.
"""
function run_ampl_problem(solver :: Symbol, problem :: Symbol, dim :: Int; args...)
  problem_s = string(problem)
  nlp = AmplModel("$problem_s.nl")
  # Objective scaling not yet available.
  stats = run_solver(solver, nlp; args...)
  amplmodel_finalize(nlp)
  return stats
end


"""
    run_solver(solver :: Symbol, nlp :: AbstractNLPModel; kwargs...)

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
function run_solver(solver :: Symbol, nlp :: AbstractNLPModel; args...)
  solver_f = eval(solver)
  args = Dict(args)
  skip = haskey(args, :skipif) ? pop!(args, :skipif) : x -> false
  skip(nlp) && throw(SkipException())

  # Julia nonsense
  optimal = false
  f = 0.0
  gNorm = 0.0
  status = "fail"
  try
    (x, f, gNorm, iter, optimal, tired, status) = solver_f(nlp; args...)
  catch e
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
