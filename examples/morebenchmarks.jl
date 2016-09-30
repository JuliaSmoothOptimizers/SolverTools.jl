# Benchmark two solvers on a set of small problems and profile.
using Optimize

# Common setup
n = 100
jump_probs = filter(name -> name != :OptimizationProblems && name != :sbrybnd, names(OptimizationProblems))
ampl_prob_dir = "/home/local/USHERBROOKE/dusj1701/Documents/Recherche/TR-ARC/UnifiedImplementation/Julia/AMPLUnconstrained/"
ampl_probs = [symbol(split(p, ".")[1]) for p in filter(x -> contains(x, ".nl"), readdir(ampl_prob_dir))]

# Example 1: benchmark two solvers on a set of problems
function two_solvers()
  solvers = [:trunk, :lbfgs]
  title = @sprintf("f+g+hprod on %d problems of size about %d", length(jump_probs), n)
  #CWD = pwd()
  #cd(ampl_prob_dir)

  profiles = bmark_and_profile(solvers, ampl_probs, n, format=:ampl, title=title, skipif=model -> model.ncon != 0)
  #cd(CWD)
end

# Example 2: benchmark one solver on problems written in two modeling languages
function two_languages()
  probs = ampl_probs ∩ jump_probs
  title = @sprintf("f+g+hprod on %d problems of size about %d", length(probs), n)
  stats = Dict{Symbol, Array{Int,2}}()
  stats[:trunk_ampl] = run_problems(:trunk, probs, n, format=:ampl, skipif=model -> model.meta.ncon != 0)
  stats[:trunk_jump] = run_problems(:trunk, probs, n, format=:jump, skipif=model -> model.meta.ncon != 0)
  profile_solvers(stats, title=title)
end

# Example 3: benchmark one solver with different options on a set of problems
function solver_options()
  title = @sprintf("f+g+hprod on %d problems of size about %d", length(jump_probs), n)
  stats = Dict{Symbol, Array{Int,2}}()
  stats[:trunk] = run_problems(:trunk, jump_probs, n, format=:jump, skipif=model -> model.meta.ncon != 0, nm_itmax=5)
  stats[:trunk_monotone] = run_problems(:trunk, jump_probs, n, format=:jump, skipif=model -> model.meta.ncon != 0, monotone=true)
  profile_solvers(stats, title=title)
end
