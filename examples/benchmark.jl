# Benchmark two solvers on a set of small problems and profile.
using BenchmarkProfiles
using Optimize
using OptimizationProblems
using NLPModels
using NLPModelsJuMP
using AmplNLReader
using DataFrames
using Plots


# In the benchmark examples, the problem lists are generator expressions
# (note the parentheses); # the problems are not generated until needed.
# Don't use Arrays to avoid holding all problems in memory simultaneously.

# Common setup
n = 100
mpb_probs = filter(name -> name != :OptimizationProblems && name != :sbrybnd, names(OptimizationProblems))

probs = ["dixmaan$x" for x in 'e':'p']
mpb_probs = Symbol.(probs)
ampl_probs = [joinpath(@__DIR__, "ampl", "$p.nl") for p in probs]

# Example 1: benchmark two solvers on a set of problems
function two_solvers()
  solvers = Dict{Symbol,Function}(:trunk => trunk, :lbfgs => lbfgs)
  bmark_args = Dict{Symbol, Any}(:skipif => model -> !unconstrained(model))
  profile_args = Dict{Symbol, Any}(:title => "f+g+hprod")
  bmark_and_profile(solvers,
                    (MathProgNLPModel(eval(p)(n), name=string(p)) for p in mpb_probs),
                    bmark_args=bmark_args, profile_args=profile_args)
  png("two-solvers")
end

# Example 2: benchmark one solver on problems written in two modeling languages
function two_languages()
  stats = Dict{Symbol, DataFrame}()
  stats[:trunk_ampl] = solve_problems(trunk,
                                      (AmplModel(p) for p in ampl_probs),
                                      skipif=model -> !unconstrained(model))
  stats[:trunk_mpb] = solve_problems(trunk,
                                     (MathProgNLPModel(eval(p)(n), name=string(p)) for p in mpb_probs),
                                     skipif=model -> !unconstrained(model))
  profile_solvers(stats, title="f+g+hprod")
  png("two-languages")
end

# Example 3: benchmark one solver with different options on a set of problems
function solver_options()
  stats = Dict{Symbol, DataFrame}()
  probs = (MathProgNLPModel(eval(p)(n), name=string(p)) for p in mpb_probs)
  stats[:trunk] = solve_problems(trunk, probs, skipif=model -> !unconstrained(model), nm_itmax=5)
  stats[:trunk_nonmonotone] = solve_problems(trunk, probs, skipif=model -> !unconstrained(model), monotone=false)
  profile_solvers(stats, title="f+g+hprod")
  png("solver-options")
end

two_solvers()
two_languages()
solver_options()
