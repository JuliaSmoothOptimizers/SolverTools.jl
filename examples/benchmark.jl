# Benchmark two solvers on a set of small problems and profile.
using Optimize

n = 100
solvers = [:trunk, :lbfgs]
probs = filter(name -> name != :OptimizationProblems, names(OptimizationProblems))
title = @sprintf("f+g+hprod on %d problems of size about %d", length(probs), n)
profiles = bmark_and_profile(solvers, probs, n, format=:jump, title=title)

# Save as png
# using Plots
# png("trunk-vs-lbfgs")
