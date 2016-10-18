module Optimize

using NLPModels
using AmplNLReader
using OptimizationProblems
using LinearOperators
using Krylov
using Compat
import Compat.String

# Algorithmic components.
include("linesearch/linesearch.jl")
include("trust-region/trust-region.jl")
include("solver/solver.jl")

# Utilities.
include("bmark/run_solver.jl")
include("bmark/bmark_solvers.jl")
if Pkg.installed("MathProgBase") != nothing
  include("bmark/mpb_vs_ampl.jl")
end

end
