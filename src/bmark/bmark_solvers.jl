export bmark_solvers, profile_solvers, bmark_and_profile


"""
    bmark_solvers(solvers :: Vector{Symbol}, probs :: Vector{Symbol}, n :: Int; kwargs...)

Run a set of solvers on a set of problems.

#### Arguments
* `solvers`: a vector of solvers to which each problem should be passed
* `probs`: a vector of problem names as symbols
  (see the `format` keyword argument)
* `n`: the approximate size in which each problem should be instantiated.
  The problem size may be adjusted automatically to the nearest smaller size
  if a particular problem's size is constrained.
  This argument has no effect on certain problem formats (see `format` below).

#### Keyword arguments
* `format::Symbol` the problem format. Currently, only `:jump` and `:ampl` are supported
* any other keyword argument accepted by `run_problems()`

#### Return value
* a Dict{Symbol, Array{Int,2}} of statistics.
"""
function bmark_solvers(solvers :: Vector{Symbol}, args...; kwargs...)
  stats = Dict{Symbol, Array{Int,2}}()
  for solver in solvers
    stats[solver] = run_problems(solver, args...; kwargs...)
  end
  return stats
end

"""
    profile_solvers(stats :: Dict{Symbol, Array{Int,2}}; title :: AbstractString="")

Plot a performance profile from solver statistics.

#### Arguments
* `stats`: a dict of statistics such as obtained from `bmark_solvers()`

#### Keyword arguments
* `title`: the plot title.
"""
function profile_solvers(stats :: Dict{Symbol, Array{Int,2}};
                         title :: AbstractString="")
  performance_profile(hcat([sum(p, 2) for p in values(stats)]...),
                      collect(AbstractString, [string(s) for s in keys(stats)]),
                      title=title)
end

"""
    bmark_and_profile(args...;
                      bmark_args :: Dict{Symbol, Any}=Dict{Symbol,Any}(),
                      profile_args :: Dict{Symbol, Any}=Dict{Symbol,Any}())

Run a set of solvers on a set of problems and plot a performance profile.

#### Arguments
Any argument accepted by `bmark_solvers()`.

#### Keyword arguments
* `bmark_args`: a dict of keyword arguments accepted by `bmark_solvers()`
* `profile_args`: a dict of keyword arguments accepted by `profile_solvers()`.
"""
function bmark_and_profile(args...;
                           bmark_args :: Dict{Symbol, Any}=Dict{Symbol,Any}(),
                           profile_args :: Dict{Symbol, Any}=Dict{Symbol,Any}())
  stats = bmark_solvers(args...; bmark_args...)
  profile_solvers(stats; profile_args...)
end
