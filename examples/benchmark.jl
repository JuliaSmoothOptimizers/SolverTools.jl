# Benchmark two solvers on a set of small problems and profile.
using BenchmarkProfiles
using Optimize
using OptimizationProblems
using NLPModels
using AmplNLReader
using MiniLogging

MiniLogging.basic_config(MiniLogging.INFO; date_format="%Y-%m-%d %H:%M:%S")
optimizelogger = get_logger("optimize")

# In the benchmark examples, the problem lists are generator expressions
# (note the parentheses); # the problems are not generated until needed.
# Don't use Arrays to avoid holding all problems in memory simultaneously.

# Common setup
n = 100
mpb_probs = filter(name -> name != :OptimizationProblems && name != :sbrybnd, names(OptimizationProblems))
ampl_prob_dir = joinpath(dirname(@__FILE__), "ampl")
ampl_probs = filter(x -> contains(x, ".nl"), readdir(ampl_prob_dir))

# Example 1: benchmark two solvers on a set of problems
function two_solvers()
  solvers = [trunk, lbfgs]
  bmark_args = Dict{Symbol, Any}(:skipif => model -> !unconstrained(model))
  profile_args = Dict{Symbol, Any}(:title => "f+g+hprod")
  bmark_and_profile(solvers,
                    (MathProgNLPModel(eval(p)(n), name=string(p)) for p in mpb_probs),
                    bmark_args=bmark_args, profile_args=profile_args)
end

# Example 2: benchmark one solver on problems written in two modeling languages
function two_languages()
  probs = ampl_probs ∩ mpb_probs
  stats = Dict{Symbol, Array{Int,2}}()
  stats[:trunk_ampl] = solve_problems(trunk,
                                      (AmplModel(p) for p in ampl_probs),
                                      skipif=model -> !unconstrained(model))
  stats[:trunk_mpb] = solve_problems(trunk,
                                     (MathProgNLPModel(eval(p)(n), name=string(p)) for p in mpb_probs),
                                     skipif=model -> !unconstrained(model))
  profile_solvers(stats, title="f+g+hprod")
end

# Example 3: benchmark one solver with different options on a set of problems
function solver_options()
  stats = Dict{Symbol, Array{Int,2}}()
  probs = (MathProgNLPModel(eval(p)(n), name=string(p)) for p in mpb_probs)
  stats[:trunk] = solve_problems(trunk, probs, skipif=model -> !unconstrained(model), nm_itmax=5)
  stats[:trunk_monotone] = solve_problems(trunk, skipif=model -> !unconstrained(model), monotone=true)
  profile_solvers(stats, title="f+g+hprod")
end
