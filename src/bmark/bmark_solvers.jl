export bmark_solvers, profile_solvers, bmark_and_profile

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
  stats = Dict{Symbol, DataFrame}()
  for (name,solver) in solvers
    @info @sprintf("running %s\n", string(solver))
    stats[name] = solve_problems(solver, args...; kwargs...)
  end
  return stats
end


profile_solvers(args...; kwargs...) = error("BenchmarkProfiles is required for profiles")
bmark_and_profile(args...; kwargs...) = error("BenchmarkProfiles is required for profiles")
