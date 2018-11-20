export display_header, solve_problems, solve_problem, uncstats, constats

struct SkipException <: Exception
end

const uncstats = [:objective, :dual_feas, :neval_obj, :neval_grad, :neval_hess, :neval_hprod, :iter, :elapsed_time, :status]
const constats = [:objective, :dual_feas, :neval_obj, :neval_grad, :neval_hess, :neval_hprod, :neval_cons, :neval_jac, :neval_jprod, :neval_jtprod, :iter, :elapsed_time, :status]

"""
    display_header()

Output header for stats table.

This function is called once before the first problem solve and can be overridden to customize the display.

#### Return value
Nothing.
"""
function display_header(;colstats::Array{Symbol} = constats)
  s = statshead(colstats)
  @info @printf("%-15s  %8s  %8s  %s\n", "Name", "nvar", "ncon", s)
end


"""
display_problem_stats(nlp, stats; colstats)

Output stats for problem `nlp` after a solve.

This function is called after each problem solve and can be overridden to customize the display.

#### Arguments
* `nlp::AbstractNLPModel`: the problem just solved
* `stats::AbstractExecutionStats`: execution statistics
* `colstats::Array{Symbol}`: list of desired stats to show

#### Return value
Nothing.
"""
function display_problem_stats(nlp::AbstractNLPModel,
                               stats::AbstractExecutionStats;
                               colstats::Array{Symbol} = constats)
  s = statsline(stats, colstats)
  @info @sprintf("%-15s  %8d  %8d  %s\n",
                 nlp.meta.name, nlp.meta.nvar, nlp.meta.ncon, s)
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
* an `Array(AbstractExecutionStats, nprobs)` where `nprobs` is the number of problems
  in `problems` minus the skipped ones if `prune` is true.
"""
function solve_problems(solver :: Function, problems :: Any; prune :: Bool=true, kwargs...)
  display_header()
  nprobs = length(problems)
  solverstr = split(string(solver), ".")[end]
  stats = []
  k = 0
  for problem in problems
    try
      s = solve_problem(solver, problem; kwargs...)
      push!(stats, s)
    catch e
      isa(e, SkipException) || rethrow(e)
      prune || push!(stats, GenericExecutionStats(:unknown, problem))
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

  stats = GenericExecutionStats(:exception, nlp)
  # try
    stats = solver(nlp; args...)
  # catch e
  #   status = :msg in fieldnames(typeof(e)) ? e.msg : string(e)
  # end
  display_problem_stats(nlp, stats, colstats=colstats)
  return stats
end
