export bmark_solvers, profile_solvers,
       bmark_and_profile


optimizelogger = get_logger("optimize")
"""
    bmark_solvers(solvers :: Vector{Function}, args...; kwargs...)

Run a set of solvers on a set of problems.

#### Arguments
* `solvers`: a vector of solvers to which each problem should be passed
* other positional arguments accepted by `solve_problems()`, except for a solver name

#### Keyword arguments
Any keyword argument accepted by `solve_problems()`

#### Return value
A Dict{Symbol, AbstractExecutionStats} of statistics.
"""
function bmark_solvers(solvers :: Dict{Symbol,Function}, args...; kwargs...)
  stats = Dict{Symbol, Array{AbstractExecutionStats,1}}()
  for (name,solver) in solvers
    @info(optimizelogger,
          @printf("running %s\n", string(solver)))
    stats[name] = solve_problems(solver, args...; kwargs...)
  end
  return stats
end


profile_solvers(args...; kwargs...) = error("BenchmarkProfiles is required for profiles")
bmark_and_profile(args...; kwargs...) = error("BenchmarkProfiles is required for profiles")

@require BenchmarkProfiles begin
  """
      profile_solvers(stats :: Dict{Symbol, Array{Int,2}}; title :: String="")

  Plot a performance profile from solver statistics.

  #### Arguments
  * `stats`: a dict of statistics such as obtained from `bmark_solvers()`

  #### Keyword arguments
  * `cost`: a function `cost(::AbstractExecutionStats)` returning a positive cost
    value. Usually, time or number of function evaluations.
  * Any keyword argument accepted by `BenchmarkProfiles.performance_profile()`.

  #### Return value
  A profile as returned by `performance_profile()`.

  #### Cost functions

  The default cost function is the number of function evaluations, i.e.,

      cost = stat->sum_counters(stat.counters)

  Another commonly used option is the elapsed time:

      cost = stat->stat.elapsed_time
  """
  function profile_solvers(stats :: Dict{Symbol, Array{AbstractExecutionStats,1}};
                           cost :: Function = stat->sum_counters(stat.counters),
                           kwargs...)
    args = Dict(kwargs)
    if haskey(args, :title)
      args[:title] *= @sprintf(" (%d problems)", size(stats[first(keys(stats))], 1))
    end
    solvers = keys(stats)
    np, ns = length(stats[first(solvers)]), length(solvers)
    P = [stats[s][p].status == :first_order ? cost(stats[s][p]) : -1 for p = 1:np, s in solvers]
    BenchmarkProfiles.performance_profile(P, map(string, solvers); args...)
  end

  """
      bmark_and_profile(args...;
                        bmark_args :: Dict{Symbol, <: Any}=Dict{Symbol,Any}(),
                        profile_args :: Dict{Symbol, <: Any}=Dict{Symbol,Any}())

  Run a set of solvers on a set of problems and plot a performance profile.

  #### Arguments
  Any positional argument accepted by `bmark_solvers()`.

  #### Keyword arguments
  * `bmark_args`: a dict of keyword arguments accepted by `bmark_solvers()`
  * `profile_args`: a dict of keyword arguments accepted by `BenchmarkProfiles.performance_profile()`.

  #### Return value
  * A Dict{Symbol, Array{Int,2}} of statistics
  * a profile as returned by `performance_profile()`.

  #### Cost functions

  The default cost function is the number of function evaluations, i.e.,

      cost = stat->sum_counters(stat.counters)

  Another commonly used option is the elapsed time:

      cost = stat->stat.elapsed_time
  """
  @compat function bmark_and_profile(args...; cost :: Function = stat->sum_counters(stat.counters),
                             bmark_args :: Dict{Symbol, <: Any}=Dict{Symbol,Any}(),
                             profile_args :: Dict{Symbol, <: Any}=Dict{Symbol,Any}())
    stats = bmark_solvers(args...; bmark_args...)
    profiles = profile_solvers(stats, cost=cost; profile_args...)
    return stats, profiles
  end
end
