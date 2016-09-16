export bmark_solvers, profile_solvers, bmark_and_profile


"Run a set of solvers on a set of problems. Return a dict of statistics."
function bmark_solvers(solvers :: Vector{Symbol}, probs :: Vector{Symbol}, n :: Int;
                       format :: Symbol=:jump, kwargs...)
  stats = Dict{Symbol, Array{Int,2}}()
  for solver in solvers
    stats[solver] = run_problems(solver, probs, n, format=format; kwargs...)
  end
  return stats
end

"Plot a performance profile from a dict of statistics such as obtained from `bmark_solvers`."
function profile_solvers(stats :: Dict{Symbol, Array{Int,2}};
                         title :: AbstractString="")
  performance_profile(hcat([sum(p, 2) for p in values(stats)]...),
                      collect(AbstractString, [string(s) for s in keys(stats)]),
                      title=title)
end

"Run a set of solvers on a set of problems and plot a performance profile."
function bmark_and_profile(args...; kwargs...)
  stats = bmark_solvers(args...; kwargs...)
  profile_solvers(stats)
end
