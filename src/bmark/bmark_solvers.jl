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
A Dict{Symbol, Array{Int,2}} of statistics.
"""
function bmark_solvers(solvers :: Vector{Function}, args...; kwargs...)
  stats = Dict{Symbol, Array{Int,2}}()
  for solver in solvers
    @printf("running %s\n", string(solver))
    stats[Symbol(solver)] = solve_problems(solver, args...; kwargs...)
  end
  return stats
end


profile_solvers(args...; kwargs...) = error("BenchmarkProfiles is required for profiles")

@require BenchmarkProfiles begin
  """
      profile_solvers(stats :: Dict{Symbol, Array{Int,2}}; title :: String="")

  Plot a performance profile from solver statistics.

  #### Arguments
  * `stats`: a dict of statistics such as obtained from `bmark_solvers()`

  #### Keyword arguments
  Any keyword argument accepted by `BenchmarkProfiles.performance_profile()`.

  #### Return value
  A profile as returned by `performance_profile()`.
  """
  function profile_solvers{T,n}(stats :: Dict{Symbol, Array{T,n}}; kwargs...)
    args = Dict(kwargs)
    if haskey(args, :title)
      args[:title] *= @sprintf(" (%d problems)", size(stats[first(keys(stats))], 1))
    end
    BenchmarkProfiles.performance_profile(hcat([sum(p, 2) for p in values(stats)]...),
                                          collect(String, [string(s) for s in keys(stats)]); args...)
  end

  """
      bmark_and_profile(args...;
                        bmark_args :: Dict{Symbol, Any}=Dict{Symbol,Any}(),
                        profile_args :: Dict{Symbol, Any}=Dict{Symbol,Any}())

  Run a set of solvers on a set of problems and plot a performance profile.

  #### Arguments
  Any positional argument accepted by `bmark_solvers()`.

  #### Keyword arguments
  * `bmark_args`: a dict of keyword arguments accepted by `bmark_solvers()`
  * `profile_args`: a dict of keyword arguments accepted by `BenchmarkProfiles.performance_profile()`.

  #### Return value
  * A Dict{Symbol, Array{Int,2}} of statistics
  * a profile as returned by `performance_profile()`.
  """
  function bmark_and_profile(args...;
                             bmark_args :: Dict{Symbol, Any}=Dict{Symbol,Any}(),
                             profile_args :: Dict{Symbol, Any}=Dict{Symbol,Any}())
    stats = bmark_solvers(args...; bmark_args...)
    profiles = profile_solvers(stats; profile_args...)
    return stats, profiles
  end
end
