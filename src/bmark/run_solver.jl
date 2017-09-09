export display_header, solve_problems, solve_problem, uncstats, constats

type SkipException <: Exception
end

const uncstats = [:obj, :dual_feas, :neval_obj, :neval_grad, :neval_hess, :neval_hprod, :iter, :elapsed_time, :status]
const constats = [:obj, :dual_feas, :primal_feas, :neval_obj, :neval_grad, :neval_hess, :neval_hprod, :neval_cons, :neval_jac, :neval_jprod, :neval_jtprod, :iter, :elapsed_time, :status]

"""`
"""
function display_header(;colstats::Array{Symbol} = constats)
  s = statshead(colstats)
  @printf("%-15s  %8s  %8s  %s\n", "Name", "nvar", "ncon", s)
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
* an `Array(ExecutionStats, nprobs)` where `nprobs` is the number of problems
  in `problems` minus the skipped ones if `prune` is true.
"""
function solve_problems(solver :: Function, problems :: Any; prune :: Bool=true, kwargs...)
  display_header()
  nprobs = length(problems)
  verbose = nprobs ≤ 1
  stats = []
  k = 0
  for problem in problems
    try
      s = solve_problem(solver, problem, verbose=verbose; kwargs...)
      push!(stats, s)
    catch e
      isa(e, SkipException) || rethrow(e)
      prune || push!(stats, ExecutionStats(:unknown))
    finally
      finalize(problem)
    end
  end
  return stats
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
function solve_problem(solver :: Function, nlp :: AbstractNLPModel; colstats::Array{Symbol} = constats, kwargs...)
  args = Dict(kwargs)
  skip = haskey(args, :skipif) ? pop!(args, :skipif) : x -> false
  skip(nlp) && throw(SkipException())

  stats = ExecutionStats(:exception)
  try
    stats = solver(nlp; args...)
    # if nlp.scale_obj
    #   f /= nlp.scale_obj_factor
    #   gNorm /= nlp.scale_obj_factor
    # end
  catch e
    println(e)
  end

  # Remove prefix.name -> name
  name = split(nlp.meta.name, ".")[end]

  s = statsline(stats, colstats)
  @printf("%-15s  %8d  %8s  %s\n", name, nlp.meta.nvar, nlp.meta.ncon, s)
  return stats
end
