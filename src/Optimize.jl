module Optimize

using JuMP
using NLPModels
using AmplNLReader
using OptimizationProblems
using LinearOperators
using Krylov
using Profiles

# Algorithmic components.
include("linesearch/linesearch.jl")
include("trust-region/trust-region.jl")
include("solver/solver.jl")

# Utilities.
include("bmark/run_solver.jl")
include("bmark/bmark_solvers.jl")
include("bmark/jump_vs_ampl.jl")
include("bmark/dercheck.jl")

end
