export solve_problems

using DataFrames

"""
    solve_problems(solver :: Function, problems :: Any; kwargs...)

Apply a solver to a set of problems.

#### Arguments
* `solver`: the function name of a solver
* `problems`: the set of problems to pass to the solver, as an iterable of
  `AbstractNLPModel`.  It is recommended to use a generator expression (necessary for
  CUTEst problems).

#### Keyword arguments
* `logger::AbstractLogger`: logger used to show the statistics. Recommended `NullLogger`
  for nothing or `ConsoleLogger`. (default: `NullLogger`).
* `skipif::Function`: function to be applied to a problem and return whether to skip it
  (default: `x->false`)
* `prune`: do not include skipped problems in the final statistics (default: `true`)
* any other keyword argument to be passed to the solvers

#### Return value
* a `DataFrame` where each line is a problem, minus the skipped ones if `prune` is true.
"""
function solve_problems(solver :: Function, problems :: Any;
                        logger :: AbstractLogger=NullLogger(),
                        skipif :: Function=x->false,
                        colstats :: Array{Symbol,1} = [:name, :nvar, :ncon, :status],
                        prune :: Bool=true, kwargs...)
  nprobs = length(problems)
  solverstr = split(string(solver), ".")[end]

  counter_fields = collect(fieldnames(Counters))
  ncounters = length(counter_fields)
  types = [Int; String;   Int;   Int;  Symbol;    Float64;       Float64;   Int;    Float64; fill(Int, ncounters)]
  names = [:id;  :name; :nvar; :ncon; :status; :objective; :elapsed_time; :iter; :dual_feas;                    counter_fields]
  stats = DataFrame(types, names, 0)

  with_logger(logger) do
    # TODO: Improve logging after #74
    @info join(string.(colstats), "  ")
  end

  for (id,problem) in enumerate(problems)
    problem_info = [id; problem.meta.name; problem.meta.nvar; problem.meta.ncon]
    if skipif(problem)
      prune || push!(stats, [problem_info; :exception; Inf; Inf; 0; Inf; fill(0, ncounters)])
      finalize(problem)
      continue
    end
    try
      s = solver(problem; kwargs...)
      push!(stats, [problem_info; s.status; s.objective; s.elapsed_time; s.iter; s.dual_feas;
                    [getfield(s.counters, f) for f in counter_fields]])
    catch e
      push!(stats, [problem_info; :exception; Inf; Inf; 0; Inf;
                    fill(0, ncounters)])
    finally
      finalize(problem)
    end
    with_logger(logger) do
      @info join(string.(Vector(stats[end,colstats])), "  ")
    end
  end
  return stats
end
