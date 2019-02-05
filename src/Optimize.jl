module Optimize
__precompile__(false)

using NLPModels, NLPModelsJuMP, LinearOperators, Krylov, Requires
using LinearAlgebra
using Printf, Logging

# Auxiliary.
include("auxiliary/bounds.jl")
include("stats/stats.jl")

# Algorithmic components.
include("linesearch/linesearch.jl")
include("trust-region/trust-region.jl")
include("solver/solver.jl")

# Utilities.
include("bmark/run_solver.jl")
include("bmark/bmark_solvers.jl")

function __init__()
  @require BenchmarkProfiles = "ecbce9bc-3e5e-569d-9e29-55181f61f8d0" include("bmark/bmark_and_profile.jl")
end

end
