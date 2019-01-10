"""
    profile_solvers(stats :: Dict{Symbol, Array{Int,2}}; title :: String="")

Plot a performance profile from solver statistics.

#### Arguments
* `stats`: a dict of statistics such as obtained from `bmark_solvers()`

#### Keyword arguments
* `cost`: a function which is applied to each row of the solver statistics DataFrame
  returning a positive cost value.
  Some useful options:
  - `cost=row -> row.elapsed_time` for the elapsed time
  - `cost=row -> row.neval_obj` for the number of objective evaluations
  - `cost=row -> sum(row[f] for f in fieldnames(Counters))` for the total number of
    function evaluations
* `success_flags`: a vector of success flags.
* Any keyword argument accepted by `BenchmarkProfiles.performance_profile()`.

#### Return value
A profile as returned by `performance_profile()`.

#### Cost functions

The default cost function is the number of function evaluations, i.e.,

    cost = stat->sum_counters(stat.counters)

Another commonly used option is the elapsed time:

    cost = stat->stat.elapsed_time
"""
function profile_solvers(stats :: Dict{Symbol, DataFrame};
                         cost :: Function = row->sum(row[f] for f in fieldnames(Counters)),
                         success_flags :: Array{Symbol} = collect(keys(STATUSES)),
                         # TODO compare objective and feasibility
                         kwargs...)
  args = Dict(kwargs)
  if haskey(args, :title)
    args[:title] *= @sprintf(" (%d problems)", size(stats[first(keys(stats))], 1))
  end
  solvers = collect(keys(stats))
  P = convert(Matrix{Float64}, hcat([[row.status in success_flags ? cost(row) : Inf for
                                      row in eachrow(stats[s])] for s in solvers]...))
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
function bmark_and_profile(args...; cost :: Function = row->sum(row[f] for f in fieldnames(Counters)),
                           success_flags :: Array{Symbol} = collect(keys(STATUSES)),
                           bmark_args :: Dict{Symbol, <: Any}=Dict{Symbol,Any}(),
                           profile_args :: Dict{Symbol, <: Any}=Dict{Symbol,Any}())
  stats = bmark_solvers(args...; bmark_args...)
  profiles = profile_solvers(stats, cost=cost, success_flags=success_flags; profile_args...)
  return stats, profiles
end
