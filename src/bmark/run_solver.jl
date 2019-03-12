export solve_problems

"""
    solve_problems(solver :: Function, problems :: Any; kwargs...)

Apply a solver to a set of problems.

#### Arguments
* `solver`: the function name of a solver
* `problems`: the set of problems to pass to the solver, as an iterable of
  `AbstractNLPModel`.  It is recommended to use a generator expression (necessary for
  CUTEst problems).

#### Keyword arguments
* `solver_logger::AbstractLogger`: logger wrapping the solver call. (default: `NullLogger`).
* `skipif::Function`: function to be applied to a problem and return whether to skip it
  (default: `x->false`)
* `prune`: do not include skipped problems in the final statistics (default: `true`)
* any other keyword argument to be passed to the solver

#### Return value
* a `DataFrame` where each row is a problem, minus the skipped ones if `prune` is true.
"""
function solve_problems(solver :: Function, problems :: Any;
                        solver_logger :: AbstractLogger=NullLogger(),
                        skipif :: Function=x->false,
                        colstats :: Vector{Symbol} = [:name, :nvar, :ncon, :status, :elapsed_time, :objective, :dual_feas, :primal_feas],
                        info_hdr_override :: Dict{Symbol,String} = Dict{Symbol,String}(),
                        prune :: Bool=true, kwargs...)
  nprobs = length(problems)
  solverstr = split(string(solver), ".")[end]

  f_counters = collect(fieldnames(Counters))
  fnls_counters = collect(fieldnames(NLSCounters))[2:end] # Excludes :counters
  ncounters = length(f_counters) + length(fnls_counters)
  types = [Int; String;   Int;   Int;   Int;  Symbol;    Float64;       Float64;   Int;    Float64;      Float64; fill(Int, ncounters); String]
  names = [:id;  :name; :nvar; :ncon; :nequ; :status; :objective; :elapsed_time; :iter; :dual_feas; :primal_feas; f_counters; fnls_counters; :extrainfo]
  stats = DataFrame(types, names, 0)

  specific = Symbol[]

  local col_idx

  first_problem = true
  for (id,problem) in enumerate(problems)
    nequ = problem isa AbstractNLSModel ? problem.nls_meta.nequ : 0
    problem_info = [id; problem.meta.name; problem.meta.nvar; problem.meta.ncon; nequ]
    if skipif(problem)
      prune || push!(stats, [problem_info; :exception; Inf; Inf; 0; Inf; Inf;
                             fill(0, ncounters); "skipped"; fill(missing, length(specific))])
      finalize(problem)
    else
      try
        s = with_logger(solver_logger) do
          solver(problem; kwargs...)
        end
        if first_problem
          for (k,v) in s.solver_specific
            insertcols!(stats, ncol(stats)+1, k => Array{Union{typeof(v),Missing},1}())
            push!(specific, k)
          end

          col_idx = indexin(colstats, names)
          @info log_header(colstats, types[col_idx], hdr_override=info_hdr_override)

          first_problem = false
        end
        push!(stats, [problem_info; s.status; s.objective; s.elapsed_time; s.iter; s.dual_feas; s.primal_feas
                      [getfield(s.counters.counters, f) for f in f_counters];
                      [getfield(s.counters, f) for f in fnls_counters]; "";
                      [s.solver_specific[k] for k in specific]])
      catch e
        push!(stats, [problem_info; :exception; Inf; Inf; 0; Inf; Inf;
                      fill(0, ncounters); string(e); fill(missing, length(specific))])
      finally
        finalize(problem)
      end
    end
    @info log_row(stats[end,col_idx])
  end
  return stats
end
